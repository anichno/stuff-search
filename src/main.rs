use std::{io::Read, path::Path};

use anyhow::{bail, Ok, Result};
use async_openai::types::{
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ImageUrlArgs,
    ResponseFormat, ResponseFormatJsonSchema,
};
use clap::Parser;
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Ingest multiple photos of items, get descriptions of them, and downscale image
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to ingest root
    #[arg(short, long)]
    ingest: String,

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
struct AllItems(Vec<ItemInfo>);

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
        .model("gpt-4o")
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

fn downscale_image_and_save(photo: &[u8], path: &Path) {
    let image = image::load_from_memory(photo).unwrap();

    // Resize image
    let (width, height) = (image.width(), image.height());
    let (new_width, new_height) = calculate_new_dimensions(width, height, 1024);
    let resized_img =
        image.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    resized_img.save(path).unwrap();
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

    // Check that ingest path has expected folders
    let ingest_path = std::path::Path::new(&args.ingest);
    let markdown_path = ingest_path.join("markdown");
    let photo_path = ingest_path.join("photos");

    if !markdown_path.exists() || !photo_path.exists() {
        bail!("Not correct ingest path");
    }

    // Setup tera templates
    let mut tera = tera::Tera::default();
    tera.add_raw_template("item.md", include_str!("item_template.md"))
        .unwrap();

    // open zip archive
    let zip_fname = std::path::Path::new(&args.zip);
    let zip_file = std::fs::File::open(zip_fname).unwrap();

    let mut archive = zip::ZipArchive::new(zip_file).unwrap();

    // OpenAI Client
    let client = async_openai::Client::new();

    // let mut all_items = Vec::new();
    for i in 0..archive.len() {
        info!("Processing {} of {}", i + 1, archive.len());

        let photo = archive.by_index(i)?;
        let photo_name = photo.name().to_string();
        let photo_data: Vec<u8> = photo.bytes().map(|b| b.unwrap()).collect();
        let photo_b64 = base64::display::Base64Display::new(
            &photo_data,
            &base64::engine::general_purpose::STANDARD,
        )
        .to_string();

        let item_info = get_description(&client, &photo_b64).await.unwrap();
        // dbg!(item_info);
        // all_items.push(item_info);

        let markdown_name = markdown_path.join(format!(
            "{} {}",
            item_info.name,
            photo_name.replace(".jpg", ".md")
        ));

        let photo_name = format!("{}_{}", &item_info.name.replace(" ", "_"), &photo_name);
        let photo_path = photo_path.join(&photo_name);
        downscale_image_and_save(&photo_data, &photo_path);

        let contents = render_template(&tera, &photo_name, &item_info);

        std::fs::write(&markdown_name, contents).unwrap();
    }

    // std::fs::write(
    //     "testing.json",
    //     serde_json::to_string_pretty(&all_items).unwrap(),
    // )
    // .unwrap();

    Ok(())
}
