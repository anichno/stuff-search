# Stuff Search

## Overview
"Stuff Search" is a semantic search tool designed to help you locate items in your house efficiently. I have slowly acquired a lot of random items from various projects I've worked on. Because of that, when working on something new I frequently am unsure of what I might have already purchased, leading to me buying duplicates. This project is an overcomplicated solution to that problem.

In general, it works by storing statements about the objects in a vector database. Later, when I need to determine what I might already have, I can search based on meaning and concept. For example, I could search for: "something that puts threads in holes", and it would return a picture and location of my drill taps.

To not make ingest too burdensome, a user takes a single photo of each item on a simple background. Then on the "Containers" page you select the container which will hold your item (or items, if submitting a zip file). Input images will be passed to OpenAI's GPT-4o-mini vision model which will generate names and descriptions of the items. Those names and descriptions will then be embedded in the vector database for later retrieval.

For a video based overview, see: https://youtu.be/ZvqfHi6xzdI?si=gtEN__ZRWj766VC8&t=1394


## Features
- Semantic search for household items.
- Auto name and description generation via OpenAI's gpt-4o-mini
- Item and container organization via drag and drop

## Getting Started

### Prerequisites
- Docker
- An OpenAI API key (https://platform.openai.com/api-keys)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anichno/stuff-search.git
   cd stuff-search
   ```
2. Build Container:
   ```bash
   docker build -t stuff-search .
   ```
3. Setup directories for bind mounts
   ```bash
   mkdir data cache
   ```

### Basic Usage
- To start the application:
  ```bash
  docker run -it --rm -e OPENAI_API_KEY=<api_key_here> -v `pwd`/data:/data -v `pwd`/cache:/cache -p 8080:8080 stuff-search
  ```
- Alternatively fill out OPENAI_API_KEY in `docker-compose.yml`, then:
  ```bash
  docker compose up
  ```

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License
[MIT License](./LICENSE)

## Acknowledgments
- [htmx](https://htmx.org/) for simplifying web app development
- [fastembed-rs](https://github.com/Anush008/fastembed-rs) for an easy embedding generation library
    - And by proxy [fastembed](https://github.com/qdrant/fastembed), from which fastembed-rs was derived
- [mxbai-embed-large-v1](https://www.mixedbread.ai/blog/mxbai-embed-large-v1) as the embedding model
- [sqlite-vec](https://github.com/asg017/sqlite-vec) for a simple sqlite based vector database
- [Sector67](https://www.sector67.org/blog/), my friendly neighborhood maker space
