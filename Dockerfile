FROM rust:slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev clang ca-certificates

WORKDIR /build

# Empty build of only dependencies for caching
COPY /Cargo.toml /Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release

COPY / ./
RUN touch src/main.rs && cargo build --release -F docker

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y libssl-dev ca-certificates
RUN mkdir /data && mkdir /cache

WORKDIR /app
COPY --from=builder /build/target/release/stuff-search .
COPY /assets/ ./assets/
COPY /templates/ ./templates/
ENV RUST_LOG=info

ENTRYPOINT [ "/app/stuff-search" ]
