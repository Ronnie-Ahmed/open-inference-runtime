// use std::collections::HashMap;
// use std::error::Error;
use tokio;
mod client;
mod tokenizer;
use client::*;
// pub use websocket_server::start_ws_server;
mod models;
// use futures::Future;
// use models::*;
// use reqwest::Client;
// use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
// use tokio_stream::{self as stream};
// use tokio_tungstenite::tungstenite::Message;
use futures::SinkExt;
use futures::StreamExt;
use warp::ws::Message;
use warp::Filter;

#[tokio::main]
async fn main() {
    // Configurations
    let triton_url = "http://localhost:8000/v2";
    // let model_name = "densenet_onnx";
    let model_path = PathBuf::from("/home/ronnie/open-inference-runtime/extract");

    // Create Triton client
    let client = match TritonClient::new(triton_url, model_path).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Failed to create Triton client: {:?}", e);
            return;
        }
    };

    // Generate dummy inputs for the model
    // let inputs = match client.generate_inputs(model_name).await {
    //     Ok(i) => i,
    //     Err(e) => {
    //         eprintln!("❌ Failed to generate inputs: {:?}", e);
    //         return;
    //     }
    // };

    // // Run inference
    // match client.run_inference(model_name, inputs).await {
    //     Ok(output) => {
    //         println!("🧠 Inference Output:\n{:?}", output);
    //     }
    //     Err(e) => {
    //         eprintln!("❌ Inference error: {:?}", e);
    //     }
    // }

    // Start WebSocket server with TritonClient
    let client_filter = warp::any().map(move || client.clone());

    let routes = warp::path("inference")
        .and(warp::ws())
        .and(client_filter)
        .map(|ws: warp::ws::Ws, client: TritonClient| {
            ws.on_upgrade(move |socket| handle_socket(socket, client))
        });

    println!("🚀 WebSocket server running on ws://127.0.0.1:3000/inference");
    println!("Try  `wscat -c ws://127.0.0.1:3000/inference` ");
    warp::serve(routes).run(([127, 0, 0, 1], 3000)).await;
}

use std::sync::Arc;
use tokio::sync::Mutex;

async fn handle_socket(ws: warp::ws::WebSocket, client: TritonClient) {
    let (ws_tx, mut ws_rx) = ws.split();

    // Wrap ws_tx in an Arc<Mutex<..>> so it can be shared
    let ws_sender = Arc::new(Mutex::new(ws_tx));

    // Channel to send incoming messages as String to TritonClient::run
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let request_stream = UnboundedReceiverStream::new(rx);

    // Spawn task to read from WS and send to channel
    tokio::spawn({
        let tx = tx.clone();
        async move {
            while let Some(Ok(msg)) = ws_rx.next().await {
                if msg.is_text() {
                    let _ = tx.send(msg.to_str().unwrap().to_string());
                }
            }
        }
    });

    // Define response closure to send Triton responses back to WS
    let response_closure = {
        let ws_sender = ws_sender.clone();
        move |response: String| {
            let ws_sender = ws_sender.clone();
            async move {
                let mut sender = ws_sender.lock().await;
                let _ = sender.send(Message::text(response)).await;
            }
        }
    };

    // Run the TritonClient message handler
    if let Err(e) = client.run(request_stream, response_closure).await {
        eprintln!("Error in Triton client run: {}", e);
    }
}
