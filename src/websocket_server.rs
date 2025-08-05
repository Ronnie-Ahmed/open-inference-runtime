use warp::Filter;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;
use std::sync::Arc;
use crate::client::TritonClient;

pub async fn start_ws_server(triton:Arc<TritonClient>){
    let triton_filter=warp::any().map(move || triton.clone());
    let route = warp::path("inference")
        .and(warp::ws())
        .and(triton_filter)
        .map(|ws: warp::ws::Ws, triton| {
            ws.on_upgrade(move |socket| handle_connection(socket, triton))
        });

        warp::serve(route).run(([0,0,0,0],3000)).await;

}

async fn handle_connection(ws: warp::ws::WebSocket, triton: Arc<TritonClient>) {
    let (mut tx, mut rx) = ws.split();
    
    while let Some(Ok(msg)) = rx.next().await {
        if msg.is_text() {
            let text = msg.to_str().unwrap();

            let inputs = match serde_json::from_str(text) {
                Ok(inputs) => inputs,
                Err(e) => {
                    tx.send(warp::ws::Message::text(format!("JSON error: {}", e))).await.ok();
                    continue;
                }
            };

            let response = match triton.run_inference(inputs).await {
                Ok(result) => result.to_string(),
                Err(e) => format!("Inference Error: {}", e),
            };

            tx.send(warp::ws::Message::text(response)).await.ok();
        }
    }
}
