use std::{
    collections::HashMap,
    fmt::{Debug, Write},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::StatusCode,
    response::{Html, Response},
    routing::{delete, get, post},
    Form, Router,
};
use minijinja::context;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tower_http::limit::RequestBodyLimitLayer;
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

impl Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState").finish()
    }
}

#[derive(Debug, Deserialize)]
struct CreateContainer {
    new_container_name: String,
    parent_container_id: i64,
}

#[derive(Debug, Deserialize)]
struct EditItem {
    new_name: String,
    new_location: String,
    new_description: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv()?;

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{}=debug", env!("CARGO_CRATE_NAME")).into()),
        )
        .with(tracing_forest::ForestLayer::default())
        .init();

    let db = Arc::new(Mutex::new(database::Database::init()?));
    let importer = Arc::new(Mutex::new(import::Importer::new(db.clone()).await));
    let shared_state = Arc::new(AppState {
        database: db,
        importer,
    });

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/page/search", get(serve_search))
        .route("/search", post(search))
        .route("/container/{id}", get(container))
        .route("/container/{id}/create", get(container_create_child))
        .route("/container/{id}/rename", get(get_container_rename))
        .route("/container/{id}/rename", post(handle_container_rename))
        .route(
            "/container/{id}/rename/cancel",
            get(get_container_rename_cancel),
        )
        .route("/container/create", post(create_container))
        .route("/modal/upload/{id}", get(modal_upload))
        .route("/upload", post(upload))
        .route("/modal/item/{id}/show", get(modal_item_show))
        .route("/model/item/{id}/edit", get(get_modal_item_edit))
        .route("/model/item/{id}/edit", post(handle_modal_item_edit))
        .route("/item/{i}", delete(delete_item_unconfirmed))
        .route("/item/{i}/confirm", delete(delete_item))
        .route("/images/small/{id}/small.jpg", get(small_photo))
        .route("/images/large/{id}/large.jpg", get(large_photo))
        .layer(DefaultBodyLimit::max(usize::MAX))
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

#[tracing::instrument]
async fn serve_search() -> Html<String> {
    Html(
        TEMPLATES
            .get_template("search.html")
            .unwrap()
            .render(context!())
            .unwrap(),
    )
}

#[tracing::instrument]
async fn search(
    State(state): State<Arc<AppState>>,
    Form(query): Form<HashMap<String, String>>,
) -> Html<String> {
    let results = if let Some(query) = query.get("search") {
        match state.database.lock().unwrap().query(&query) {
            Ok(results) => results,
            Err(e) => {
                error!("{}", e);
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    Html(
        TEMPLATES
            .get_template("search.html")
            .unwrap()
            .eval_to_state(context!(results))
            .unwrap()
            .render_block("query_results")
            .unwrap(),
    )
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

async fn container(State(state): State<Arc<AppState>>, Path(id): Path<i64>) -> Html<String> {
    let Ok(containers) = state.database.lock().unwrap().get_container_tree() else {
        return Html(String::from("Failed to retrieve containers"));
    };

    let Ok(items) = state.database.lock().unwrap().get_container_items(id) else {
        return Html(String::from("Failed to retrieve items"));
    };

    Html(
        TEMPLATES
            .get_template("containers/containers.html")
            .unwrap()
            .render(context!(container => containers, results => items, active_node_id => id))
            .unwrap(),
    )
}

async fn container_create_child(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Html<String> {
    let Ok(containers) = state.database.lock().unwrap().get_container_tree() else {
        return Html(String::from("Failed to retrieve containers"));
    };

    let Ok(items) = state.database.lock().unwrap().get_container_items(id) else {
        return Html(String::from("Failed to retrieve items"));
    };

    Html(
        TEMPLATES
            .get_template("containers/containers.html")
            .unwrap()
            .render(context!(container => containers, results => items, active_node_id => id, add_child => true))
            .unwrap(),
    )
}

async fn create_container(
    State(state): State<Arc<AppState>>,
    Form(payload): Form<CreateContainer>,
) -> Html<String> {
    state
        .database
        .lock()
        .unwrap()
        .add_child_container(&payload.new_container_name, payload.parent_container_id)
        .unwrap();

    let Ok(containers) = state.database.lock().unwrap().get_container_tree() else {
        return Html(String::from("Failed to retrieve containers"));
    };

    let Ok(items) = state
        .database
        .lock()
        .unwrap()
        .get_container_items(payload.parent_container_id)
    else {
        return Html(String::from("Failed to retrieve items"));
    };

    Html(
        TEMPLATES
            .get_template("containers/containers.html")
            .unwrap()
            .render(context!(container => containers, results => items, active_node_id => payload.parent_container_id))
            .unwrap(),
    )
}

async fn get_container_rename(
    State(state): State<Arc<AppState>>,
    Path(container_id): Path<i64>,
) -> Html<String> {
    let Ok(container_name) = state
        .database
        .lock()
        .unwrap()
        .get_container_name(container_id)
    else {
        return Html(String::from("Failed to fetch container name"));
    };

    Html(
        TEMPLATES
            .get_template("containers/container_edit.html")
            .unwrap()
            .render(context!(container_name, container_id))
            .unwrap(),
    )
}

async fn get_container_rename_cancel(
    State(state): State<Arc<AppState>>,
    Path(container_id): Path<i64>,
) -> Html<String> {
    let Ok(container_name) = state
        .database
        .lock()
        .unwrap()
        .get_container_name(container_id)
    else {
        return Html(String::from("Failed to fetch container name"));
    };

    Html(
        TEMPLATES
            .get_template("containers/container_single.html")
            .unwrap()
            .render(context!(container_name, container_id))
            .unwrap(),
    )
}

async fn handle_container_rename(
    State(state): State<Arc<AppState>>,
    Path(container_id): Path<i64>,
    Form(new_container_name): Form<HashMap<String, String>>,
) -> Html<String> {
    state
        .database
        .lock()
        .unwrap()
        .set_container_name(
            &new_container_name.get("new_container_name").unwrap(),
            container_id,
        )
        .unwrap();

    let Ok(containers) = state.database.lock().unwrap().get_container_tree() else {
        return Html(String::from("Failed to retrieve containers"));
    };

    let Ok(items) = state
        .database
        .lock()
        .unwrap()
        .get_container_items(container_id)
    else {
        return Html(String::from("Failed to retrieve items"));
    };

    Html(
        TEMPLATES
            .get_template("containers/containers.html")
            .unwrap()
            .render(
                context!(container => containers, results => items, active_node_id => container_id),
            )
            .unwrap(),
    )
}

async fn modal_upload(
    State(state): State<Arc<AppState>>,
    Path(container_id): Path<i64>,
) -> Html<String> {
    let Ok(container_name) = state
        .database
        .lock()
        .unwrap()
        .get_container_name(container_id)
    else {
        return Html(String::from("Failed to retrieve container"));
    };

    Html(
        TEMPLATES
            .get_template("containers/modal_upload.html")
            .unwrap()
            .render(context!(container_name, container_id, in_progress => false))
            .unwrap(),
    )
}

async fn upload(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> Html<String> {
    let mut container_id = None;
    let mut file = None;
    let mut file_name = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or_default();

        match field_name {
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                if let Ok(bytes) = field.bytes().await {
                    file = Some(bytes.to_vec());
                }
            }
            "container" => {
                if let Ok(text) = field.text().await {
                    if let Ok(id) = text.parse::<i64>() {
                        container_id = Some(id)
                    }
                }
            }
            _ => (),
        }
    }

    if let (Some(container_id), Some(file)) = (container_id, file) {
        let Ok(container_name) = state
            .database
            .lock()
            .unwrap()
            .get_container_name(container_id)
        else {
            return Html(String::from("Failed to retrieve container"));
        };
        if state
            .importer
            .lock()
            .unwrap()
            .add_to_queue(import::ImportRequest {
                source: file_name.unwrap_or(String::from("Unknown Filename")),
                file,
                target_container: container_id,
            })
            .is_err()
        {
            return Html(String::from("Failed to upload file to queue"));
        }

        return Html(
            TEMPLATES
                .get_template("containers/modal_upload.html")
                .unwrap()
                .render(context!(container_name, container_id, in_progress => true))
                .unwrap(),
        );
    }

    Html(String::from(
        "<script>bootstrap.Modal.getInstance(document.getElementById('modals-here')).hide()</script>",
    ))
}

async fn modal_item_show(
    State(state): State<Arc<AppState>>,
    Path(item_id): Path<i64>,
) -> Html<String> {
    let Ok(item) = state.database.lock().unwrap().get_item(item_id) else {
        return Html(String::from("Failed to retrieve item"));
    };

    Html(
        TEMPLATES
            .get_template("items/modal_display.html")
            .unwrap()
            .render(context!(item_id => item.id, item_name => item.name, item_location => item.container_name, item_description => item.description))
            .unwrap(),
    )
}

async fn get_modal_item_edit(
    State(state): State<Arc<AppState>>,
    Path(item_id): Path<i64>,
) -> Html<String> {
    let Ok(item) = state.database.lock().unwrap().get_item(item_id) else {
        return Html(String::from("Failed to retrieve item"));
    };

    Html(
        TEMPLATES
            .get_template("items/modal_edit.html")
            .unwrap()
            .render(context!(item_id => item.id, item_name => item.name, item_location => item.container_name, item_description => item.description))
            .unwrap(),
    )
}

async fn handle_modal_item_edit(
    State(state): State<Arc<AppState>>,
    Path(item_id): Path<i64>,
    Form(edit_item): Form<EditItem>,
) -> Html<String> {
    state
        .database
        .lock()
        .unwrap()
        .update_item(item_id, &edit_item.new_name, &edit_item.new_description)
        .unwrap();

    Html(
            TEMPLATES
                .get_template("items/modal_display.html")
                .unwrap()
                .render(context!(item_id, item_name => edit_item.new_name, item_location => edit_item.new_location, item_description => edit_item.new_description))
                .unwrap(),
        )
}

async fn delete_item_unconfirmed(Path(item_id): Path<i64>) -> Html<String> {
    Html(
        TEMPLATES
            .get_template("items/delete_confirm_snippet.html")
            .unwrap()
            .render(context!(item_id))
            .unwrap(),
    )
}

async fn delete_item(State(state): State<Arc<AppState>>, Path(item_id): Path<i64>) -> Html<String> {
    // get item container
    let container_id = state
        .database
        .lock()
        .unwrap()
        .get_item(item_id)
        .unwrap()
        .container_id;

    // do deletion
    state.database.lock().unwrap().delete_item(item_id).unwrap();

    // return relevant container page
    let Ok(containers) = state.database.lock().unwrap().get_container_tree() else {
        return Html(String::from("Failed to retrieve containers"));
    };

    let Ok(items) = state
        .database
        .lock()
        .unwrap()
        .get_container_items(container_id)
    else {
        return Html(String::from("Failed to retrieve items"));
    };

    Html(
        TEMPLATES
            .get_template("containers/containers.html")
            .unwrap()
            .render(
                context!(container => containers, results => items, active_node_id => container_id),
            )
            .unwrap(),
    )
}
