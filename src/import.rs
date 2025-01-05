use std::{
    io::Read,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use async_openai::types::{
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ImageUrlArgs,
    ResponseFormat, ResponseFormatJsonSchema,
};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::info;

use crate::database::Database;

pub struct ImportRequest {
    pub source: String,
    pub file: Vec<u8>,
    pub target_container: i64,
}

pub struct Importer {
    db_conn: Arc<Mutex<Database>>,
    queue: UnboundedSender<(i64, ImportRequest)>,
}

impl Importer {
    pub async fn new(db: Arc<Mutex<Database>>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(process_queue(db.clone(), rx));
        Self {
            db_conn: db,
            queue: tx,
        }
    }

    pub fn add_to_queue(&self, request: ImportRequest) -> Result<()> {
        let log_id = self.db_conn.lock().unwrap().log_new_import(
            &request.source,
            "Added to queue",
            request.target_container,
        )?;
        Ok(self.queue.send((log_id, request))?)
    }
}

async fn process_queue(db: Arc<Mutex<Database>>, mut rx: UnboundedReceiver<(i64, ImportRequest)>) {
    // OpenAI Client
    let client = async_openai::Client::new();

    'outer: while let Some((log_id, request)) = rx.recv().await {
        info!("New file in queue");
        db.lock()
            .unwrap()
            .update_import(log_id, "Starting")
            .unwrap();
        let mut cursor = std::io::Cursor::new(request.file);
        let mut image_queue = Vec::new();

        // try to process as zip
        if let Ok(mut archive) = zip::ZipArchive::new(&mut cursor) {
            for i in 0..archive.len() {
                info!("Extracting {} of {}", i + 1, archive.len());
                db.lock()
                    .unwrap()
                    .update_import(
                        log_id,
                        &format!("Extracting {} of {}", i + 1, archive.len()),
                    )
                    .unwrap();

                let Ok(photo) = archive.by_index(i) else {
                    db.lock()
                        .unwrap()
                        .cancel_import(log_id, Some("invalid file in zip"))
                        .unwrap();
                    continue 'outer;
                };
                let photo_data: Vec<u8> = photo.bytes().map(|b| b.unwrap()).collect();
                let Ok(image) = image::load_from_memory(&photo_data) else {
                    db.lock()
                        .unwrap()
                        .cancel_import(log_id, Some("file not an image"))
                        .unwrap();
                    continue 'outer;
                };
                image_queue.push(image);
            }
        } else {
            // try to process as image
            let Ok(image) = image::load_from_memory(&cursor.into_inner()) else {
                db.lock()
                    .unwrap()
                    .cancel_import(log_id, Some("file not an image"))
                    .unwrap();
                continue 'outer;
            };
            image_queue.push(image);
        }

        let queue_len = image_queue.len();
        for (i, image) in image_queue.into_iter().enumerate() {
            info!("Processing image {} of {}", i + 1, queue_len);
            db.lock()
                .unwrap()
                .update_import(
                    log_id,
                    &format!("Processing image {} of {}", i + 1, queue_len),
                )
                .unwrap();

            let mut photo_data = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut photo_data),
                    image::ImageFormat::Jpeg,
                )
                .unwrap();

            let photo_b64 = base64::display::Base64Display::new(
                &photo_data,
                &base64::engine::general_purpose::STANDARD,
            )
            .to_string();

            let client = client.clone();
            // let item_info =
            // tokio::spawn(async move { get_description(&client, &photo_b64).await.unwrap() });
            let photo_resized_large = downscale_image(&image, 1024);
            let photo_resized_small = downscale_image(&image, 512);
            // let item_info = item_info.await.expect("OpenAI request failed");
            let item_info = ItemInfo {
                name: "testname".to_string(),
                description: "testdescription".to_string(),
            };

            db.lock()
                .unwrap()
                .insert_item(
                    &item_info.name,
                    &item_info.description,
                    &photo_resized_small,
                    &photo_resized_large,
                )
                .unwrap();
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ItemInfo {
    name: String,
    description: String,
}

async fn get_description(
    client: &async_openai::Client<async_openai::config::OpenAIConfig>,
    photo_b64: &str,
) -> Result<ItemInfo> {
    let schema = json!({
        "type": "object",
        "properties": {
        "name": {
            "type": "string",
            "description": "The name of the object."
        },
        "description": {
            "type": "string",
            "description": "A brief description of what the object is."
        }
        },
        "required": [
        "name",
        "description"
        ],
        "additionalProperties": false
    });

    let response_format = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: None,
            name: "Item_description".into(),
            schema: Some(schema),
            strict: Some(true),
        },
    };

    let  request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o-mini")
        .max_tokens(600_u32)
        .messages([ChatCompletionRequestSystemMessage::from(
            "You are a helpful item identifier and describer. You always respond in valid JSON.",
        )
        .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(vec![
                ChatCompletionRequestMessageContentPartTextArgs::default()
                    .text("Please give a short name for this object.")
                    .build()?
                    .into(),
                ChatCompletionRequestMessageContentPartTextArgs::default()
                    .text("Please give a full description of what you see in this image. Do not mention the background or any human hands.")
                    .build()?
                    .into(),
                ChatCompletionRequestMessageContentPartImageArgs::default()
                    .image_url(
                        ImageUrlArgs::default()
                            .url(format!("data:image/jpeg;base64,{}",photo_b64))
                            .detail(async_openai::types::ImageDetail::High)
                            .build()?,
                    )
                    .build()?
                    .into(),
            ])
            .build()?
            .into()]).response_format(response_format)
        .build()?;

    let response = client.chat().create(request).await?;

    let item_info: ItemInfo =
        serde_json::from_str(&response.choices[0].message.content.clone().unwrap())?;

    Ok(item_info)
}

fn calculate_new_dimensions(width: u32, height: u32, max_dimension: u32) -> (u32, u32) {
    if width > height {
        // Landscape orientation or square
        let new_width = max_dimension;
        let new_height = (max_dimension as f64 * height as f64 / width as f64).round() as u32;
        (new_width, new_height)
    } else {
        // Portrait orientation
        let new_height = max_dimension;
        let new_width = (max_dimension as f64 * width as f64 / height as f64).round() as u32;
        (new_width, new_height)
    }
}

fn downscale_image(image: &DynamicImage, max_dim: u32) -> Vec<u8> {
    // Resize image
    let (width, height) = (image.width(), image.height());
    let (new_width, new_height) = calculate_new_dimensions(width, height, max_dim);
    let resized_img =
        image.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    let mut data = Vec::new();
    resized_img
        .write_to(
            &mut std::io::Cursor::new(&mut data),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    data
}
