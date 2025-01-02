use anyhow::Result;
use fastembed::TextEmbedding;
use log::{debug, info};
use zerocopy::IntoBytes;

#[derive(Debug)]
pub struct ItemResult {
    id: i64,
    pub name: String,
    pub description: String,
    pub small_photo: Vec<u8>,
    pub similarity: f64,
    pub containers: Vec<(String, Option<String>)>,
}

pub struct Database {
    conn: rusqlite::Connection,
    model: TextEmbedding,
    temp_container_id: i64,
}

impl Database {
    pub fn init() -> Result<Self> {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = if let Ok(conn) = rusqlite::Connection::open_with_flags(
            "./storage.db",
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ) {
            conn
        } else {
            let conn = rusqlite::Connection::open("./storage.db")?;

            conn.execute(
                "CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[1024])",
                [],
            )?;

            conn.execute(
                r#"CREATE TABLE "Items" (
                            "id"	INTEGER NOT NULL UNIQUE,
                            "name"	TEXT NOT NULL,
                            "description"	TEXT NOT NULL,
                            "small_photo"	BLOB NOT NULL,
                            "large_photo"	BLOB NOT NULL,
                            "embedding_id"	INTEGER NOT NULL UNIQUE,
                            "contained_by"  INTEGER NOT NULL,
                            PRIMARY KEY("id" AUTOINCREMENT)
                        )"#,
                [],
            )?;

            conn.execute(
                r#"CREATE TABLE "containers" (
                            "id"	INTEGER NOT NULL UNIQUE,
                            "contained_by"	INTEGER,
                            PRIMARY KEY("id" AUTOINCREMENT)
                        );"#,
                [],
            )?;

            conn.execute(
                r#"CREATE TABLE "import_log" (
                            "id"	INTEGER NOT NULL UNIQUE,
                            "source"	TEXT NOT NULL,
                            "status"	TEXT NOT NULL,
                            "target_container"	INTEGER NOT NULL,
                            PRIMARY KEY("id" AUTOINCREMENT)
                        );"#,
                [],
            )?;

            conn.execute(
                r#"CREATE INDEX "idx_embedding_id" ON "Items" (
                            "embedding_id"
                        );"#,
                [],
            )?;

            conn.execute(
                r#"CREATE INDEX "idx_item_container" ON "Items" (
                            "contained_by"
                        );"#,
                [],
            )?;

            conn.execute(
                r#"CREATE INDEX "idx_container_container" ON "containers" (
                            "contained_by"
                        );"#,
                [],
            )?;

            conn.execute(r#"INSERT INTO containers(name) VALUES ("TEMP")"#, [])?;

            conn
        };

        let (sqlite_version, vec_version): (String, String) = conn
            .query_row("select sqlite_version(), vec_version()", [], |x| {
                anyhow::Result::Ok((x.get(0).unwrap(), x.get(1).unwrap()))
            })
            .unwrap();

        info!("sqlite_version={sqlite_version}, vec_version={vec_version}");

        let model = TextEmbedding::try_new(fastembed::InitOptions::new(
            fastembed::EmbeddingModel::MxbaiEmbedLargeV1,
        ))?;

        let temp_container_id = conn.query_row(
            r#"SELECT id from containers where name = "TEMP""#,
            [],
            |row| Ok(row.get(0)?),
        )?;

        Ok(Self {
            conn,
            model,
            temp_container_id,
        })
    }

    pub fn insert_item(
        &self,
        name: &str,
        description: &str,
        small_photo: &[u8],
        large_photo: &[u8],
    ) -> Result<()> {
        let embedding = self
            .model
            .embed(vec![format!("{}\n{}", name, description)], None)?
            .pop()
            .unwrap();

        let mut stmt = self
            .conn
            .prepare("INSERT INTO vec_items(embedding) VALUES (?)")?;
        stmt.execute(rusqlite::params![embedding.as_bytes()])?;
        let last_id = self.conn.last_insert_rowid();

        self.conn.prepare(r#"INSERT INTO 
                                    Items(name, description, small_photo, large_photo, embedding_id, contained_by)
                                    VALUES (?,?,?,?,?,?)"#)?
                                    .execute(rusqlite::params![name, description, small_photo.as_bytes(), large_photo.as_bytes(), last_id, self.temp_container_id])?;

        Ok(())
    }

    pub fn query(&self, query: &str) -> Result<Vec<ItemResult>> {
        let start = std::time::Instant::now();
        let query_embedding = self
            .model
            .embed(
                vec![format!(
                    "Represent this sentence for searching relevant passages: {query}"
                )],
                None,
            )?
            .pop()
            .unwrap();

        debug!(
            "Query embedding generated in: {}ms",
            std::time::Instant::now().duration_since(start).as_millis()
        );

        let embedding_result: Vec<(i64, f64)> = self
            .conn
            .prepare(
                r#"
                    SELECT
                        rowid,
                        distance
                    FROM vec_items
                    WHERE embedding MATCH ?1
                    ORDER BY distance
                    LIMIT 30
                    "#,
            )?
            .query_map([query_embedding.as_bytes()], |r| {
                anyhow::Result::Ok((r.get(0)?, r.get(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut item_results = Vec::new();
        for (embedding_id, distance) in embedding_result {
            let (id, name, description, small_photo, contained_by): (i64, String, String, Vec<u8>, i64) = self
                .conn
                .prepare(
                    "SELECT id, name, description, small_photo, contained_by FROM Items WHERE embedding_id = ?",
                )?
                .query_row([embedding_id], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?
                    ))
                })?;

            let mut contained_by = Some(contained_by);
            let mut storage_chain = Vec::new();
            while let Some(holder) = contained_by {
                let (name, location, new_holder): (String, Option<String>, Option<i64>) = self
                    .conn
                    .prepare("SELECT name, location, contained_by from containers where id = ?")?
                    .query_row([holder], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;

                storage_chain.push((name, location));
                contained_by = new_holder;
            }

            let item = ItemResult {
                id,
                name,
                description,
                small_photo,
                similarity: distance,
                containers: storage_chain,
            };
            item_results.push(item);
        }

        Ok(item_results)
    }
}
