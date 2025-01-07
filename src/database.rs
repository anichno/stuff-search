use std::{collections::HashMap, error::Error, fmt::Debug};

use anyhow::{bail, Result};
use fastembed::TextEmbedding;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use zerocopy::IntoBytes;

#[derive(Debug, Serialize)]
pub struct ItemResult {
    pub id: i64,
    pub name: String,
    pub description: String,
    // pub small_photo: Vec<u8>,
    pub similarity: f64,
    pub container_name: String,
    pub container_id: i64,
}

#[derive(Debug, Serialize)]
pub struct ContainerTree {
    pub id: i64,
    pub name: String,
    pub location: Option<String>,
    pub containers: Vec<ContainerTree>,
}

pub struct Database {
    conn: rusqlite::Connection,
    model: TextEmbedding,
}

impl Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database").finish()
    }
}

impl Database {
    #[tracing::instrument]
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
                            "name"	TEXT NOT NULL,
                            "location"	TEXT,
                            "contained_by"	INTEGER,
                            PRIMARY KEY("id" AUTOINCREMENT)
                        )"#,
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

            conn.execute(r#"INSERT INTO containers(id, name) VALUES (1, "ROOT")"#, [])?;

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

    #[tracing::instrument]
    pub fn insert_item(
        &self,
        name: &str,
        description: &str,
        small_photo: &[u8],
        large_photo: &[u8],
        contained_by: i64,
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
                                    .execute(rusqlite::params![name, description, small_photo.as_bytes(), large_photo.as_bytes(), last_id, contained_by])?;

        Ok(())
    }

    #[tracing::instrument]
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

        #[derive(Debug, Deserialize)]
        struct QueryResult {
            id: i64,
            name: String,
            description: String,
            contained_by: i64,
            container_name: String,
        }

        let mut item_results = Vec::new();
        for (embedding_id, distance) in embedding_result {
            let result = self
                .conn
                .prepare(
                    "SELECT a.id, a.name, a.description, a.contained_by, b.name as container_name FROM Items a JOIN containers b ON a.contained_by = b.id WHERE a.embedding_id = ?",
                )?.query_row([embedding_id], |row| Ok(serde_rusqlite::from_row::<QueryResult>(row).unwrap()))?;

            let item = ItemResult {
                id: result.id,
                name: result.name,
                description: result.description,
                similarity: distance,
                container_name: result.container_name,
                container_id: result.contained_by,
            };
            item_results.push(item);
        }

        Ok(item_results)
    }

    #[tracing::instrument]
    pub fn log_new_import(&self, source: &str, status: &str, target_container: i64) -> Result<i64> {
        let mut stmt = self
            .conn
            .prepare("INSERT INTO import_log(source, status, target_container) VALUES (?,?,?)")?;
        stmt.execute(rusqlite::params![
            source.as_bytes(),
            status.as_bytes(),
            target_container
        ])?;

        Ok(self.conn.last_insert_rowid())
    }

    #[tracing::instrument]
    pub fn cancel_import(&self, import_id: i64, reason: Option<&str>) -> Result<()> {
        let reason = reason.unwrap_or("FAILED");
        let mut stmt = self
            .conn
            .prepare(r#"UPDATE import_log SET status = ? where id = ?"#)?;
        stmt.execute(rusqlite::params![reason, import_id])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn update_import(&self, import_id: i64, status: &str) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare(r#"UPDATE import_log SET status = ? where id = ?"#)?;
        stmt.execute(rusqlite::params![status.as_bytes(), import_id])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn get_small_image(&self, item_id: i64) -> Result<Vec<u8>> {
        let image: Vec<u8> = self
            .conn
            .prepare("SELECT small_photo FROM Items where id = ?")?
            .query_row([item_id], |row| Ok(row.get(0)?))?;

        Ok(image)
    }

    #[tracing::instrument]
    pub fn get_large_image(&self, item_id: i64) -> Result<Vec<u8>> {
        let image: Vec<u8> = self
            .conn
            .prepare("SELECT large_photo FROM Items where id = ?")?
            .query_row([item_id], |row| Ok(row.get(0)?))?;

        Ok(image)
    }

    #[tracing::instrument]
    pub fn get_container_tree(&self) -> Result<ContainerTree> {
        let mut containers: HashMap<i64, Vec<ContainerRow>> = HashMap::new();
        let mut root = None;
        self.conn
            .prepare("SELECT id, name, location, contained_by FROM containers")?
            .query_map([], |row| {
                Ok(ContainerRow {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    location: row.get(2)?,
                    contained_by: row.get(3)?,
                })
            })?
            .for_each(|row| {
                if let Ok(row) = row {
                    if row.id == 1 {
                        root = Some(row);
                    } else {
                        if let Some(contained_by) = row.contained_by {
                            containers.entry(contained_by).or_default().push(row)
                        }
                    }
                }
            });

        let Some(root) = root else {
            bail!("Failed to find ROOT container");
        };

        let mut root = ContainerTree {
            id: root.id,
            name: root.name,
            location: root.location,
            containers: Vec::new(),
        };

        fill_tree(&mut root, &mut containers);

        Ok(root)
    }

    #[tracing::instrument]
    pub fn get_container_items(&self, container_id: i64) -> Result<Vec<ItemResult>> {
        let mut item_results = Vec::new();
        self.conn
            .prepare("SELECT id, name, description FROM Items WHERE contained_by = ?")?
            .query_map([container_id], |row| {
                Ok(ItemResult {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                    similarity: 0.0,
                    container_name: String::new(),
                    container_id,
                })
            })?
            .for_each(|row| {
                if let Ok(row) = row {
                    item_results.push(row);
                }
            });

        Ok(item_results)
    }

    #[tracing::instrument]
    pub fn get_container_name(&self, container_id: i64) -> Result<String> {
        let name: String = self
            .conn
            .prepare("SELECT name FROM containers WHERE id = ?")?
            .query_row([container_id], |row| Ok(row.get(0)?))?;

        Ok(name)
    }

    #[tracing::instrument]
    pub fn set_container_name(&self, container_name: &str, container_id: i64) -> Result<()> {
        self.conn
            .prepare("UPDATE containers SET name = ? WHERE id = ?")?
            .execute(rusqlite::params![container_name, container_id])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn get_container_parent(&self, container_id: i64) -> Result<i64> {
        let parent: i64 = self
            .conn
            .prepare("SELECT contained_by FROM containers WHERE id = ?")?
            .query_row([container_id], |row| Ok(row.get(0)?))?;

        Ok(parent)
    }

    #[tracing::instrument]
    pub fn get_container_children(&self, container_id: i64) -> Result<Vec<i64>> {
        let mut children = Vec::new();
        for child_row in serde_rusqlite::from_rows::<i64>(
            self.conn
                .prepare("SELECT id FROM containers WHERE contained_by = ?")?
                .query([container_id])?,
        ) {
            if let Ok(child) = child_row {
                children.push(child);
            }
        }

        Ok(children)
    }

    #[tracing::instrument]
    /// This will recursively delete all containers and items
    pub fn delete_container(&self, container_id: i64) -> Result<()> {
        let mut containers = vec![container_id];
        while let Some(cur_container_id) = containers.pop() {
            for item in self.get_container_items(cur_container_id).unwrap() {
                self.delete_item(item.id).unwrap();
            }

            containers.extend(self.get_container_children(cur_container_id).unwrap());

            // Do deletion of this container, now that it has no items and we know its children
            self.conn
                .prepare("DELETE FROM containers WHERE id = ?")?
                .execute([cur_container_id])?;
        }

        Ok(())
    }

    #[tracing::instrument]
    pub fn add_child_container(&self, name: &str, parent_id: i64) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("INSERT INTO containers(name, contained_by) VALUES (?,?)")?;
        stmt.execute(rusqlite::params![name, parent_id,])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn get_item(&self, item_id: i64) -> Result<ItemResult> {
        #[derive(Debug, Deserialize)]
        struct QueryResult {
            id: i64,
            name: String,
            description: String,
            contained_by: i64,
            container_name: String,
        }

        let result = self
                .conn
                .prepare(
                    "SELECT a.id, a.name, a.description, a.contained_by, b.name as container_name FROM Items a JOIN containers b ON a.contained_by = b.id WHERE a.id = ?",
                )?.query_row([item_id], |row| Ok(serde_rusqlite::from_row::<QueryResult>(row).unwrap()))?;

        Ok(ItemResult {
            id: result.id,
            name: result.name,
            description: result.description,
            similarity: 0.0,
            container_name: result.container_name,
            container_id: result.contained_by,
        })
    }

    #[tracing::instrument]
    pub fn update_item(&self, item_id: i64, item_name: &str, item_description: &str) -> Result<()> {
        // new embedding
        let embedding = self
            .model
            .embed(vec![format!("{}\n{}", item_name, item_description)], None)?
            .pop()
            .unwrap();

        // get related embedding_id
        let embedding_id: i64 = self
            .conn
            .prepare("SELECT embedding_id FROM Items where id = ?")?
            .query_row([item_id], |row| Ok(row.get(0)?))?;

        // update embedding
        self.conn
            .prepare("UPDATE vec_items SET embedding = ? where rowid = ?")?
            .execute(rusqlite::params![embedding.as_bytes(), embedding_id])?;

        // update item record
        self.conn
            .prepare("UPDATE Items SET name = ?, description = ? WHERE id = ?")?
            .execute(rusqlite::params![item_name, item_description, item_id])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn delete_item(&self, item_id: i64) -> Result<()> {
        // get related embedding_id
        let embedding_id: i64 = self
            .conn
            .prepare("SELECT embedding_id FROM Items where id = ?")?
            .query_row([item_id], |row| Ok(row.get(0)?))?;

        // delete embedding
        self.conn
            .prepare("DELETE FROM vec_items where rowid = ?")?
            .execute(rusqlite::params![embedding_id])?;

        // delete item
        self.conn
            .prepare("DELETE FROM Items where id = ?")?
            .execute(rusqlite::params![item_id])?;

        Ok(())
    }
}

#[derive(Debug)]
struct ContainerRow {
    id: i64,
    name: String,
    location: Option<String>,
    contained_by: Option<i64>,
}

#[tracing::instrument]
fn fill_tree(cur_node: &mut ContainerTree, contained_by_map: &mut HashMap<i64, Vec<ContainerRow>>) {
    if let Some(containers) = contained_by_map.remove(&cur_node.id) {
        cur_node
            .containers
            .extend(containers.into_iter().map(|c| ContainerTree {
                id: c.id,
                name: c.name,
                location: c.location,
                containers: Vec::new(),
            }));
        cur_node
            .containers
            .sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

        for sub_container in cur_node.containers.iter_mut() {
            fill_tree(sub_container, contained_by_map);
        }
    }
}
