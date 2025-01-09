use std::{collections::HashMap, fmt::Debug, sync::Mutex};

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
    conn: std::sync::Mutex<rusqlite::Connection>,
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
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let db_name = if std::env::var("DEMO").is_ok() {
            if std::fs::exists("storage.demo.db").unwrap() {
                std::fs::copy("storage.demo.db", "storage.demo.db.tmp").unwrap();
                "storage.demo.db.tmp"
            } else {
                "storage.demo.db"
            }
        } else {
            "storage.db"
        };

        #[cfg(feature = "docker")]
        let base_path = std::path::Path::new("/data");
        #[cfg(not(feature = "docker"))]
        let base_path = std::path::Path::new(".");

        let conn = if let Ok(conn) = rusqlite::Connection::open_with_flags(
            base_path.join(db_name),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ) {
            conn
        } else {
            let conn = rusqlite::Connection::open(base_path.join(db_name))?;

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
                r#"CREATE TABLE "embedding_to_item" (
                            "id"	INTEGER NOT NULL UNIQUE,
                            "embedding_id"	INTEGER NOT NULL,
                            "item_id"	INTEGER NOT NULL,
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

            conn.execute(
                r#"CREATE INDEX "idx_embedding_to_item_item_id" ON "embedding_to_item" (
                            "item_id"
                        )"#,
                [],
            )?;

            conn.execute(
                r#"CREATE INDEX "idx_embedding_to_item_embedding_id" ON "embedding_to_item" (
                            "embedding_id"
                        )"#,
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

        let fastembed_opts =
            fastembed::InitOptions::new(fastembed::EmbeddingModel::MxbaiEmbedLargeV1);

        #[cfg(feature = "docker")]
        let fastembed_opts = fastembed_opts.with_cache_dir(std::path::PathBuf::from("/cache"));

        let model = TextEmbedding::try_new(fastembed_opts)?;

        Ok(Self {
            conn: Mutex::new(conn),
            model,
        })
    }

    #[tracing::instrument(skip(description_statements))]
    fn insert_embeddings(
        &self,
        name: &str,
        description_statements: &[&str],
        item_id: i64,
    ) -> Result<()> {
        let mut embedding_docs = vec![name];
        embedding_docs.extend_from_slice(description_statements);

        let full_description = description_statements.join("\n");
        embedding_docs.push(&full_description);
        let embeddings = self.model.embed(embedding_docs, None)?;

        let conn = self.conn.lock().unwrap();
        for embedding in embeddings {
            conn.prepare("INSERT INTO vec_items(embedding) VALUES (?)")?
                .execute(rusqlite::params![embedding.as_bytes()])?;
            let embedding_id = conn.last_insert_rowid();

            conn.execute(
                "INSERT INTO embedding_to_item(embedding_id, item_id) VALUES(?,?)",
                [embedding_id, item_id],
            )?;
        }

        Ok(())
    }

    #[tracing::instrument(skip(small_photo, large_photo))]
    pub fn insert_item(
        &self,
        name: &str,
        description: &[String],
        small_photo: &[u8],
        large_photo: &[u8],
        contained_by: i64,
    ) -> Result<()> {
        let item_id = {
            let conn: std::sync::MutexGuard<'_, rusqlite::Connection> = self.conn.lock().unwrap();
            conn.prepare(
                r#"INSERT INTO 
                        Items(name, description, small_photo, large_photo, contained_by)
                        VALUES (?,?,?,?,?)"#,
            )?
            .execute(rusqlite::params![
                name,
                description.join("\n"),
                small_photo.as_bytes(),
                large_photo.as_bytes(),
                contained_by
            ])?;

            conn.last_insert_rowid()
        };

        self.insert_embeddings(
            name,
            &description
                .iter()
                .map(String::as_str)
                .collect::<Vec<&str>>(),
            item_id,
        )?;

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

        let conn = self.conn.lock().unwrap();
        let embedding_result: Vec<(i64, f64)> = conn
            .prepare(
                r#"
                    SELECT
                        rowid,
                        distance
                    FROM vec_items
                    WHERE embedding MATCH ?1
                    ORDER BY distance
                    LIMIT 100
                    "#,
            )?
            .query_map([query_embedding.as_bytes()], |r| {
                anyhow::Result::Ok((r.get(0)?, r.get(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut item_ids = Vec::new();
        let mut item_hits = HashMap::new();
        for (embedding_id, _distance) in embedding_result {
            let item_id: i64 = conn.query_row(
                "SELECT item_id FROM embedding_to_item WHERE embedding_id = ?",
                [embedding_id],
                |row| Ok(row.get(0)),
            )??;
            if !item_hits.contains_key(&item_id) {
                item_ids.push(item_id);
            }
            *item_hits.entry(item_id).or_insert(0) += 1;
        }
        let mut item_hits: Vec<(i64, i64)> = item_hits.iter().map(|(k, v)| (*k, *v)).collect();
        item_hits.sort_by(|a, b| a.1.cmp(&b.1).reverse());

        #[derive(Debug, Deserialize)]
        struct QueryResult {
            id: i64,
            name: String,
            description: String,
            contained_by: i64,
            container_name: String,
        }

        let mut item_results = Vec::new();
        for item_id in item_ids {
            let result: QueryResult = conn.query_row("SELECT a.id, a.name, a.description, a.contained_by, b.name as container_name FROM Items a JOIN containers b ON a.contained_by = b.id WHERE a.id = ?", [item_id], |row| Ok(serde_rusqlite::from_row(row).unwrap()))?;
            item_results.push(ItemResult {
                id: result.id,
                name: result.name,
                description: result.description,
                similarity: 0.0,
                container_name: result.container_name,
                container_id: result.contained_by,
            });
        }

        Ok(item_results)
    }

    #[tracing::instrument]
    pub fn log_new_import(&self, source: &str, status: &str, target_container: i64) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("INSERT INTO import_log(source, status, target_container) VALUES (?,?,?)")?;
        stmt.execute(rusqlite::params![
            source.as_bytes(),
            status.as_bytes(),
            target_container
        ])?;

        Ok(conn.last_insert_rowid())
    }

    #[tracing::instrument]
    pub fn cancel_import(&self, import_id: i64, reason: Option<&str>) -> Result<()> {
        let reason = reason.unwrap_or("FAILED");
        self.conn.lock().unwrap().execute(
            r#"UPDATE import_log SET status = ? where id = ?"#,
            rusqlite::params![reason, import_id],
        )?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn update_import(&self, import_id: i64, status: &str) -> Result<()> {
        self.conn.lock().unwrap().execute(
            r#"UPDATE import_log SET status = ? where id = ?"#,
            rusqlite::params![status.as_bytes(), import_id],
        )?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn get_small_image(&self, item_id: i64) -> Result<Vec<u8>> {
        let image: Vec<u8> = self
            .conn
            .lock()
            .unwrap()
            .prepare("SELECT small_photo FROM Items where id = ?")?
            .query_row([item_id], |row| Ok(row.get(0)))??;

        Ok(image)
    }

    #[tracing::instrument]
    pub fn get_large_image(&self, item_id: i64) -> Result<Vec<u8>> {
        let image: Vec<u8> = self
            .conn
            .lock()
            .unwrap()
            .prepare("SELECT large_photo FROM Items where id = ?")?
            .query_row([item_id], |row| Ok(row.get(0)))??;

        Ok(image)
    }

    #[tracing::instrument]
    pub fn get_container_tree(&self) -> Result<ContainerTree> {
        let mut containers: HashMap<i64, Vec<ContainerRow>> = HashMap::new();
        let mut root = None;
        self.conn
            .lock()
            .unwrap()
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
                    } else if let Some(contained_by) = row.contained_by {
                        containers.entry(contained_by).or_default().push(row)
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
            .lock()
            .unwrap()
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
            .lock()
            .unwrap()
            .prepare("SELECT name FROM containers WHERE id = ?")?
            .query_row([container_id], |row| Ok(row.get(0)))??;

        Ok(name)
    }

    #[tracing::instrument]
    pub fn set_container_name(&self, container_name: &str, container_id: i64) -> Result<()> {
        self.conn
            .lock()
            .unwrap()
            .prepare("UPDATE containers SET name = ? WHERE id = ?")?
            .execute(rusqlite::params![container_name, container_id])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn get_container_parent(&self, container_id: i64) -> Result<i64> {
        let parent: i64 = self
            .conn
            .lock()
            .unwrap()
            .prepare("SELECT contained_by FROM containers WHERE id = ?")?
            .query_row([container_id], |row| Ok(row.get(0)))??;

        Ok(parent)
    }

    #[tracing::instrument]
    pub fn get_container_children(&self, container_id: i64) -> Result<Vec<i64>> {
        let mut children = Vec::new();
        for child_row in serde_rusqlite::from_rows::<i64>(
            self.conn
                .lock()
                .unwrap()
                .prepare("SELECT id FROM containers WHERE contained_by = ?")?
                .query([container_id])?,
        )
        .flatten()
        {
            children.push(child_row);
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
                .lock()
                .unwrap()
                .prepare("DELETE FROM containers WHERE id = ?")?
                .execute([cur_container_id])?;
        }

        Ok(())
    }

    #[tracing::instrument]
    pub fn add_child_container(&self, name: &str, parent_id: i64) -> Result<()> {
        self.conn.lock().unwrap().execute(
            "INSERT INTO containers(name, contained_by) VALUES (?,?)",
            rusqlite::params![name, parent_id,],
        )?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn move_container(&self, container_source_id: i64, container_target_id: i64) -> Result<()> {
        self.conn
            .lock()
            .unwrap()
            .prepare("UPDATE containers SET contained_by = ? where id = ?")?
            .execute([container_target_id, container_source_id])?;

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
                .conn.lock().unwrap()
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
        self.delete_embeddings_for_item(item_id)?;

        let description_statements: Vec<&str> = item_description.split("\n").collect();
        self.insert_embeddings(item_name, &description_statements, item_id)?;

        // update item record
        self.conn
            .lock()
            .unwrap()
            .prepare("UPDATE Items SET name = ?, description = ? WHERE id = ?")?
            .execute(rusqlite::params![item_name, item_description, item_id])?;

        Ok(())
    }

    #[tracing::instrument]
    fn delete_embeddings_for_item(&self, item_id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        // get related embedding_ids
        let mut embedding_ids = Vec::new();
        for embedding_id in serde_rusqlite::from_rows::<i64>(
            conn.prepare("SELECT embedding_id FROM embedding_to_item where item_id = ?")?
                .query([item_id])?,
        ) {
            if let Ok(embedding_id) = embedding_id {
                embedding_ids.push(embedding_id);
            }
        }

        for embedding_id in embedding_ids {
            // delete embedding
            conn.prepare("DELETE FROM vec_items where rowid = ?")?
                .execute(rusqlite::params![embedding_id])?;
        }

        conn.prepare("DELETE FROM embedding_to_item where item_id = ?")?
            .execute(rusqlite::params![item_id])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn delete_item(&self, item_id: i64) -> Result<()> {
        self.delete_embeddings_for_item(item_id)?;

        // delete item
        self.conn
            .lock()
            .unwrap()
            .prepare("DELETE FROM Items where id = ?")?
            .execute(rusqlite::params![item_id])?;

        Ok(())
    }

    #[tracing::instrument]
    pub fn move_item(&self, item_id: i64, container_id: i64) -> Result<()> {
        self.conn
            .lock()
            .unwrap()
            .prepare("UPDATE Items SET contained_by = ? where id = ?")?
            .execute([container_id, item_id])?;

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
