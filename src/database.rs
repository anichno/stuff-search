use anyhow::Result;
use fastembed::TextEmbedding;
use log::info;
use zerocopy::IntoBytes;

#[derive(Debug)]
pub struct ItemResult {
    id: i64,
    pub name: String,
    pub description: String,
    pub small_photo: Vec<u8>,
    pub similarity: f64,
}

pub struct Database {
    conn: rusqlite::Connection,
    model: TextEmbedding,
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
                            PRIMARY KEY("id" AUTOINCREMENT)
                        )"#,
                [],
            )?;

            conn.execute(
                r#"CREATE INDEX "idx_embedding_id" ON "Items" (
                            "embedding_id"
                        )"#,
                [],
            )?;

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

        Ok(Self { conn, model })
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

        self.conn.prepare("INSERT INTO Items(name, description, small_photo, large_photo, embedding_id) VALUES (?,?,?,?,?)")?.execute(rusqlite::params![name, description, small_photo.as_bytes(), large_photo.as_bytes(), last_id])?;

        Ok(())
    }

    pub fn query(&self, query: &str) -> Result<Vec<ItemResult>> {
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
                    LIMIT 3
                    "#,
            )?
            .query_map([query_embedding.as_bytes()], |r| {
                anyhow::Result::Ok((r.get(0)?, r.get(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut item_results = Vec::new();
        for (embedding_id, distance) in embedding_result {
            let item: ItemResult = self
                .conn
                .prepare(
                    "SELECT id, name, description, small_photo FROM Items WHERE embedding_id = ?",
                )?
                .query_row([embedding_id], |row| {
                    Ok(ItemResult {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        description: row.get(2)?,
                        small_photo: row.get(3)?,
                        similarity: distance,
                    })
                })?;
            item_results.push(item);
        }

        Ok(item_results)
    }
}
