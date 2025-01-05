use std::{
    fmt::Write,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use axum::{
    body::Bytes,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{Html, Response},
    routing::{get, post},
    Router,
};
use clap::builder::Str;
use tera::Tera;
use tracing::{debug, error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod database;
mod import;

lazy_static::lazy_static! {
    pub static ref TEMPLATES: Tera = {
        let tera = match Tera::new("templates/*") {
            Ok(t) => t,
            Err(e) => {
                println!("Parsing error(s): {}", e);
                ::std::process::exit(1);
            }
        };
        tera
    };
}

struct AppState {
    database: Arc<Mutex<database::Database>>,
    importer: Arc<Mutex<import::Importer>>,
}

// /// Ingest multiple photos of items, get descriptions of them, and downscale image
// #[derive(Parser, Debug)]
// #[command(version, about, long_about = None)]
// struct Args {
//     // /// Path to ingest root
//     // #[arg(short, long)]
//     // ingest: String,
//     /// Path to zip file of photos
//     #[arg(short, long)]
//     zip: String,
// }

// fn render_template(tera: &tera::Tera, photo_name: &str, item: &ItemInfo) -> String {
//     #[derive(Debug, Serialize)]
//     struct ItemContext {
//         photo_name: String,
//         name: String,
//         description: String,
//     }

//     let context = ItemContext {
//         photo_name: photo_name.to_string(),
//         name: item.name.clone(),
//         description: item.description.clone(),
//     };

//     tera.render("item.md", &tera::Context::from_serialize(&context).unwrap())
//         .unwrap()
// }

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv()?;

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{}=debug", env!("CARGO_CRATE_NAME")).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // let args = Args::parse();

    // Setup tera templates
    let mut tera = tera::Tera::default();
    tera.add_raw_template("item.md", include_str!("item_template.md"))
        .unwrap();

    let db = Arc::new(Mutex::new(database::Database::init()?));
    let importer = Arc::new(Mutex::new(import::Importer::new(db.clone()).await));
    let shared_state = Arc::new(AppState {
        database: db,
        importer,
    });

    // let zip_file = std::fs::read("Photos-001/20241230_192801.jpg")?;
    // importer.lock().unwrap().add_to_queue(ImportRequest {
    //     source: "testing".to_string(),
    //     file: zip_file,
    //     target_container: 1,
    // })?;

    // loop {
    //     std::thread::sleep(std::time::Duration::from_secs(1));
    // }

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/page/search", get(serve_search))
        .route("/page/containers", get(serve_containers))
        .route("/modal_upload", get(serve_modal_upload))
        .route("/search", post(search))
        .route("/containers", get(container_tree))
        .route("/container/{id}/items", get(get_container_items))
        // .route("/upload", post(upload))
        .route("/images/small/{id}/small.jpg", get(small_photo))
        .route("/images/large/{id}/large.jpg", get(large_photo))
        .with_state(Arc::clone(&shared_state))
        .nest_service("/assets", tower_http::services::ServeDir::new("assets"));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3001")
        .await
        .unwrap();
    info!("Listening");
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

async fn serve_index() -> Response {
    Response::new(
        tokio::fs::read("templates/index.html")
            .await
            .unwrap()
            .into(),
    )
}

async fn serve_search() -> Response {
    Response::new(
        tokio::fs::read("templates/search.html")
            .await
            .unwrap()
            .into(),
    )
}

async fn serve_containers() -> Response {
    Response::new(
        tokio::fs::read("templates/containers.html")
            .await
            .unwrap()
            .into(),
    )
}

async fn serve_modal_upload() -> Response {
    Response::new(
        tokio::fs::read("templates/modal_upload.html")
            .await
            .unwrap()
            .into(),
    )
}

async fn search(State(state): State<Arc<AppState>>, query: String) -> Html<String> {
    match state.database.lock().unwrap().query(&query) {
        Ok(results) => {
            let mut context = tera::Context::new();
            context.insert("results", &results);

            Html(TEMPLATES.render("query_results.html", &context).unwrap())
        }
        Err(e) => Html(e.to_string()),
    }
}

async fn small_photo(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<Bytes, StatusCode> {
    match state.database.lock().unwrap().get_small_image(id) {
        Ok(image) => Ok(image.into()),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

async fn large_photo(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<Bytes, StatusCode> {
    match state.database.lock().unwrap().get_large_image(id) {
        Ok(image) => Ok(image.into()),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

async fn container_tree(State(state): State<Arc<AppState>>) -> Html<String> {
    let Ok(containers) = state.database.lock().unwrap().get_container_tree() else {
        return Html(String::from("Failed to retrieve containers"));
    };

    let mut context = tera::Context::new();
    context.insert("container", &containers);

    Html(TEMPLATES.render("container_tree.html", &context).unwrap())
}

async fn get_container_items(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Html<String> {
    match state.database.lock().unwrap().get_container_items(id) {
        Ok(results) => {
            let mut context = tera::Context::new();
            context.insert("results", &results);

            Html(TEMPLATES.render("query_results.html", &context).unwrap())
        }
        Err(e) => Html(e.to_string()),
    }
}

// async fn upload(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> Html<String> {
//     let mut container = None;
//     let mut file = None;

//     while let Some(field) = multipart.next_field().await.unwrap() {
//         let field_name = field.name().unwrap_or_default();

//         match field_name {
//             "file" => file = Some(field.bytes().await.unwrap().to_vec()),
//             "container" => container = Some(field.text().await.unwrap()),
//             _ => (),
//         }
//     }

//     if let (Some(container), Some(file)) = (container, file) {
//         // Get container
//     }
//     todo!()
// }
