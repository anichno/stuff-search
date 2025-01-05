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
use minijinja::context;
use tracing::{debug, error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod database;
mod import;

lazy_static::lazy_static! {
    pub static ref TEMPLATES: minijinja::Environment<'static> = {
        let mut env = minijinja::Environment::new();
        env.set_loader(minijinja::path_loader("templates"));

        env

    };
}

struct AppState {
    database: Arc<Mutex<database::Database>>,
    importer: Arc<Mutex<import::Importer>>,
}

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

async fn serve_search() -> Html<String> {
    Html(
        TEMPLATES
            .get_template("search.html")
            .unwrap()
            .render(context!())
            .unwrap(),
    )
}

async fn serve_containers() -> Html<String> {
    Html(
        TEMPLATES
            .get_template("containers.html")
            .unwrap()
            .render(context!())
            .unwrap(),
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
        Ok(results) => Html(
            TEMPLATES
                .get_template("search.html")
                .unwrap()
                .eval_to_state(context!(results))
                .unwrap()
                .render_block("query_results")
                .unwrap(),
        ),
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

    Html(
        TEMPLATES
            .get_template("containers.html")
            .unwrap()
            .eval_to_state(context!(container => containers))
            .unwrap()
            .render_block("container_tree")
            .unwrap(),
    )
}

async fn get_container_items(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Html<String> {
    match state.database.lock().unwrap().get_container_items(id) {
        Ok(results) => Html(
            TEMPLATES
                .get_template("containers.html")
                .unwrap()
                .eval_to_state(context!(results))
                .unwrap()
                .render_block("query_results")
                .unwrap(),
        ),
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
