use std::{
    io::{BufReader, BufWriter, Read, Seek, Write},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{bail, Result};
use async_openai::types::{
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ImageUrlArgs,
    ResponseFormat, ResponseFormatJsonSchema,
};
use image::DynamicImage;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{error, info};

use crate::database::Database;

pub struct ImportRequest {
    pub source: String,
    pub file: std::fs::File,
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

struct ImageFileReader(Mutex<std::fs::File>);

impl ImageFileReader {
    fn new(mut file: std::fs::File) -> Result<Self> {
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let photo_file_buffered = BufReader::new(file.try_clone().unwrap());

        if let Ok(image_reader) = image::ImageReader::new(photo_file_buffered).with_guessed_format()
        {
            if image_reader.decode().is_ok() {
                file.seek(std::io::SeekFrom::Start(0)).unwrap();
                return Ok(Self(Mutex::new(file)));
            } else {
                bail!("Failed to decode");
            }
        } else {
            bail!("Failed to build image reader");
        }
    }
    fn to_image(&self) -> DynamicImage {
        let mut inner = self.0.lock().unwrap();
        inner.seek(std::io::SeekFrom::Start(0)).unwrap();
        let buf_reader = BufReader::new(inner.try_clone().unwrap());
        let image = image::ImageReader::new(buf_reader)
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        inner.seek(std::io::SeekFrom::Start(0)).unwrap();

        image
    }
}

impl From<ImageFileReader> for Vec<u8> {
    fn from(value: ImageFileReader) -> Self {
        let mut photo_data = Vec::new();
        value
            .to_image()
            .to_rgb8()
            .write_to(
                &mut std::io::Cursor::new(&mut photo_data),
                image::ImageFormat::Jpeg,
            )
            .unwrap();

        photo_data
    }
}

async fn process_queue(db: Arc<Mutex<Database>>, mut rx: UnboundedReceiver<(i64, ImportRequest)>) {
    // OpenAI Client
    let client = async_openai::Client::new();

    while let Some((log_id, request)) = rx.recv().await {
        info!("New file in queue");
        db.lock()
            .unwrap()
            .update_import(log_id, "Starting")
            .unwrap();
        let mut image_queue: Vec<ImageFileReader> = Vec::new();

        // try to process as zip
        if let Ok(mut archive) = zip::ZipArchive::new(&request.file) {
            for i in 0..archive.len() {
                info!("Extracting {} of {}", i + 1, archive.len());
                if let Ok(photo) = archive.by_index(i) {
                    let photo_name = photo.name().to_owned();
                    if photo.is_file() {
                        let photo_file = tempfile::tempfile().unwrap();
                        let mut photo_file = BufWriter::new(photo_file);
                        for byte in photo.bytes() {
                            photo_file.write_all(&[byte.unwrap()]).unwrap()
                        }
                        photo_file.flush().unwrap();
                        let mut photo_file = photo_file.into_inner().unwrap();
                        photo_file.seek(std::io::SeekFrom::Start(0)).unwrap();

                        match ImageFileReader::new(photo_file) {
                            Ok(photo_file) => image_queue.push(photo_file),
                            Err(e) => error!(
                                "Encountered: {} on {} ({})",
                                e.to_string(),
                                i + 1,
                                photo_name
                            ),
                        }
                    } else {
                        error!("Entry {} is not a file ({})", i + 1, photo_name);
                    }
                } else {
                    error!("Failed to extract {}", i + 1);
                }
            }
        } else {
            // try to process as image
            match ImageFileReader::new(request.file) {
                Ok(photo_file) => image_queue.push(photo_file),
                Err(e) => error!("Single image {}", e.to_string()),
            }
        }

        let image_queue = Arc::new(image_queue);
        let resize_image_queue = image_queue.clone();
        let resize_job = tokio::task::spawn_blocking(move || {
            resize_image_queue
                .par_iter()
                .enumerate()
                .map(|(i, image_reader)| {
                    info!("Starting resize {}", i + 1);
                    let photo_resized_large = downscale_image(&image_reader, 1024);
                    let photo_resized_small = downscale_image(&image_reader, 512);
                    info!("Done resize {}", i + 1);
                    (photo_resized_small, photo_resized_large)
                })
                .collect::<Vec<(ImageFileReader, ImageFileReader)>>()
        });

        let mut openai_item_info = Vec::new();
        let image_queue_len = image_queue.len();
        for i in 0..image_queue_len {
            let client = client.clone();
            let openai_image_queue = image_queue.clone();
            openai_item_info.push(tokio::spawn(async move {
                let mut photo_data = Vec::new();
                openai_image_queue[i]
                    .to_image()
                    .to_rgb8()
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

                info!("Starting openai request {}", i + 1);
                let mut item_info = None;
                for retry in 0..10 {
                    match get_description(&client, &photo_b64).await {
                        Ok(info) => {
                            item_info = Some(info);
                            break;
                        }
                        Err(e) => error!(
                            "OpenAI request failed, retry {} of 10. Msg: {}",
                            retry + 1,
                            e.to_string()
                        ),
                    }
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }

                info!("End openai request {}", i + 1);
                item_info
            }));
        }

        let resized_results = resize_job.await.unwrap();
        for ((resized_small, resized_large), openai_info) in resized_results
            .into_iter()
            .zip(openai_item_info.into_iter())
        {
            if let Some(item_info) = openai_info.await.unwrap() {
                let resized_small: Vec<u8> = resized_small.into();
                let resized_large: Vec<u8> = resized_large.into();
                db.lock()
                    .unwrap()
                    .insert_item(
                        &item_info.name,
                        &item_info.descriptions,
                        &resized_small,
                        &resized_large,
                        request.target_container,
                    )
                    .unwrap();
            } else {
                error!("Failed to import");
            }
        }

        db.lock()
            .unwrap()
            .update_import(log_id, "Complete")
            .unwrap();
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ItemInfo {
    name: String,
    descriptions: Vec<String>,
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
        "descriptions": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "A series of statements giving a full and detailed description of what you see in this image, including all text you can read."
        }
        },
        "required": [
        "name",
        "descriptions"
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
        .max_tokens(1000_u32)
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
                    .text("Please give a full and detailed description of what you see in this image. Include all text you can read. Give the description as a series of statements. Do not mention the background or any human hands.")
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
        serde_json::from_str(&response.choices[0].message.content.clone().unwrap()).unwrap();

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

fn downscale_image(image_file: &ImageFileReader, max_dim: u32) -> ImageFileReader {
    let image = image_file.to_image();

    // Resize image
    let (width, height) = (image.width(), image.height());
    let (new_width, new_height) = calculate_new_dimensions(width, height, max_dim);
    let resized_img =
        image.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    let outfile = tempfile::tempfile().unwrap();
    let mut buffered_outfile = BufWriter::new(outfile);
    resized_img
        .to_rgb8()
        .write_to(&mut buffered_outfile, image::ImageFormat::Jpeg)
        .unwrap();

    let mut outfile = buffered_outfile.into_inner().unwrap();
    outfile.seek(std::io::SeekFrom::Start(0)).unwrap();

    ImageFileReader(Mutex::new(outfile))
}
