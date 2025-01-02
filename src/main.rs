use std::io::Read;

use anyhow::Result;
use async_openai::types::{
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ImageUrlArgs,
    ResponseFormat, ResponseFormatJsonSchema,
};
use clap::Parser;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use serde_json::json;

mod database;

/// Ingest multiple photos of items, get descriptions of them, and downscale image
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // /// Path to ingest root
    // #[arg(short, long)]
    // ingest: String,
    /// Path to zip file of photos
    #[arg(short, long)]
    zip: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ItemInfo {
    name: String,
    description: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ItemRecord {
    name: String,
    description: String,
    full_photo: Vec<u8>,
    resized_photo: Vec<u8>,
}

#[derive(Debug, Deserialize, Serialize)]
struct AllItems(Vec<ItemRecord>);

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

fn downscale_image(photo: &[u8], max_dim: u32) -> Vec<u8> {
    let image = image::load_from_memory(photo).unwrap();

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

fn render_template(tera: &tera::Tera, photo_name: &str, item: &ItemInfo) -> String {
    #[derive(Debug, Serialize)]
    struct ItemContext {
        photo_name: String,
        name: String,
        description: String,
    }

    let context = ItemContext {
        photo_name: photo_name.to_string(),
        name: item.name.clone(),
        description: item.description.clone(),
    };

    tera.render("item.md", &tera::Context::from_serialize(&context).unwrap())
        .unwrap()
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv()?;
    env_logger::init();

    let args = Args::parse();

    // Setup tera templates
    let mut tera = tera::Tera::default();
    tera.add_raw_template("item.md", include_str!("item_template.md"))
        .unwrap();

    let db = database::Database::init()?;

    // open zip archive
    let zip_fname = std::path::Path::new(&args.zip);
    let zip_file = std::fs::File::open(zip_fname).unwrap();

    let mut archive = zip::ZipArchive::new(zip_file).unwrap();

    // OpenAI Client
    let client = async_openai::Client::new();

    for i in 0..archive.len() {
        info!("Processing {} of {}", i + 1, archive.len());

        let photo = archive.by_index(i)?;
        let photo_data: Vec<u8> = photo.bytes().map(|b| b.unwrap()).collect();
        let photo_b64 = base64::display::Base64Display::new(
            &photo_data,
            &base64::engine::general_purpose::STANDARD,
        )
        .to_string();

        let client = client.clone();
        let item_info =
            tokio::spawn(async move { get_description(&client, &photo_b64).await.unwrap() });
        let photo_resized_large = downscale_image(&photo_data, 1024);
        let photo_resized_small = downscale_image(&photo_data, 128);
        let item_info = item_info.await?;

        db.insert_item(
            &item_info.name,
            &item_info.description,
            &photo_resized_small,
            &photo_resized_large,
        )?;
    }

    let results = db.query("a secret plane")?;
    for result in results {
        let containers: Vec<String> = result
            .containers
            .into_iter()
            .map(|(n, l)| {
                if let Some(loc) = l {
                    if !loc.is_empty() {
                        format!("{} ({})", n, loc)
                    } else {
                        n
                    }
                } else {
                    n
                }
            })
            .collect();
        println!(
            "name: {}\ndescription: {}\nscore: {}\ncontainers: {}\n",
            result.name,
            result.description,
            result.similarity,
            containers.join(" -> ")
        );
    }

    Ok(())
}
