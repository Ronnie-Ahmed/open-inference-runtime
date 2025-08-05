use std::collections::HashMap;
use std::error::Error;
use tokio;
mod client;
mod websocket_server;
pub use websocket_server::start_ws_server;
use client::*;
mod models;
use models::*;
use std::path::PathBuf;
use tokio_stream::{self as stream, StreamExt};

// const TRITON_URL: &str = "http://localhost:8000/v2";

// const BASE_PATH: &str = "/var/lib/cyborg/miner/current_task/";

// const model_path: &str = "/home/ronnie/Model";

// #[tokio::main]
// async fn main() -> Result<(), Box<dyn Error>> {
//     // 1️⃣ Step 1: Extract Model
//     // let archive_path = "./model/mobilenetv2-7.tar.gz";
//     // let output_folder = "/home/ronnie/server/docs/examples/model_repository/";

//     // // Create an extractor instance
//     // let extractor = ModelExtractor::new(archive_path, output_folder);

//     // extractor.extract_model()?;

//     // 2️⃣ Step 2: Load Model into Triton
//     let model_name = "densenet_onnx"; // Replace with your model's name
//     let client = TritonClient::new();

//     println!("Server live: {:?}", client.is_server_live().await.unwrap());
//     println!("-------------------------------------------");
//     println!("-------------------------------------------");
//     println!("Server ready: {:?}", client.is_server_ready().await.unwrap());
//     println!("-------------------------------------------");
//     println!("-------------------------------------------");

//     // 📜 List currently loaded models
//     let models = client.list_models().await.unwrap();
//     println!("Models Loaded: {:?}", models);
//     println!("-------------------------------------------");
//     println!("-------------------------------------------");
//     println!("🚀 Loading model into Triton...");
//     match client.load_model(model_name).await {
//         Ok(_) => println!("✅ Model successfully loaded!"),
//         Err(e) => {
//             println!("❌ Failed to load model: {:?}", e);
//             return Err(Box::new(e) as Box<dyn std::error::Error>);
//         }
//     }

//     let metadata = client.get_model_metadata(model_name).await?;
//     println!("Model Metadata: {:#?}", metadata);
//     println!("-------------------------------------------");
//     println!("-------------------------------------------");
//     println!("-------------------------------------------");
//     println!("-------------------------------------------");

//     // If user does not provide input, generate dummy data
//     // let user_provided_data: Option<HashMap<&str, (Vec<f32>, Vec<usize>)>> = None;
//     // let input_data = if user_provided_data.is_some() {
//     //     user_provided_data.unwrap()
//     // } else {
//     //     println!("⚠️ No input data provided, generating dummy data...");
//     //     client.prepare_input_data( model_name).await?
//     // };

//     // Dynamically create input data based on model metadata
//     // println!("📝 Preparing input data...");
//     // let mut input_data = HashMap::new();

//     // if let Some(inputs) = metadata["inputs"].as_array() {
//     //     for input in inputs {
//     //         let input_name = input["name"].as_str().unwrap();

//     //         // Handle dynamic dimensions (-1) gracefully
//     //         let shape = input["shape"]
//     //             .as_array()
//     //             .unwrap()
//     //             .iter()
//     //             .map(|v| {
//     //                 match v.as_i64() {
//     //                     Some(-1) => 1, // Replace -1 with 1 for dummy data
//     //                     Some(val) if val > 0 => val as usize,
//     //                     _ => panic!("❌ Unexpected dimension size"),
//     //                 }
//     //             })
//     //             .collect::<Vec<usize>>();

//     //         println!("✅ Detected Input: {} with Shape {:?}", input_name, shape);

//     //         // Create dummy data (all 0.5) matching the required shape
//     //         let num_elements: usize = shape.iter().product();
//     //         let data = vec![0.5_f32; num_elements];
//     //         input_data.insert(input_name, (data, shape));
//     //     }
//     // }

//     let mut input_data = HashMap::new();
//     let input_name = "data_0";
//     let input_shape = vec![3, 224, 224]; // Shape from model metadata
//     let num_elements: usize = input_shape.iter().product();
//     let data = vec![0.5_f32; num_elements]; // Dummy data

//     input_data.insert(input_name, (TensorData::F32(data), input_shape));

//     println!("🚀 Sending inference request...");
//     match client.infer(model_name, input_data).await {
//         Ok(result) => println!("✅ Inference Result: {:#?}", result),
//         Err(e) => println!("❌ Inference failed: {:?}", e),
//     }

//      // 3️⃣ Step 3: Fetch Model Status &
//      println!("🔍 Fetching model status...");
//      let status_model = client.get_model_status(model_name).await?;
//      println!("Model Status: {:#?}", status_model);
//      println!("-------------------------------------------");
//      println!("-------------------------------------------");
//      println!("-------------------------------------------");
//      println!("-------------------------------------------");

//     // // 5️⃣ (Optional) Unload Model
//     println!("🚀 Unloading model...");
//     client.unload_model(model_name).await?;
//     println!("✅ Model successfully unloaded!");

//     Ok(())

// }

///var/lib/cyborg/miner/current_task/model_archive.tar.gz

#[tokio::main]
async fn main() {
    // Configurations
    let triton_url = "http://localhost:8000/v2";
    let model_name = "densenet_onnx";
    let model_path = PathBuf::from("/home/ronnie/open-inference-runtime/extract");

    // Create Triton client
    let client = match TritonClient::new(triton_url, model_name, model_path.clone()).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Failed to create Triton client: {:?}", e);
            return;
        }
    };
    start_ws_server(client.clone().into()).await;

    // Create test input data
    let mut input_data: HashMap<String, TensorData> = HashMap::new();
    input_data.insert(
        "data_0".to_string(),
        TensorData::F32(vec![0.1; 3 * 224 * 224]),
    );

    client.run_inference(input_data).await.unwrap();

    // // Serialize to JSON string to simulate a real WebSocket message
    // let json_string = match serde_json::to_string(&input_data) {
    //     Ok(s) => s,
    //     Err(e) => {
    //         eprintln!("Failed to serialize input: {}", e);
    //         return;
    //     }
    // };

    // // Create a stream of one message (like WebSocket would send)
    // let request_stream = stream::iter(vec![json_string]);

    // // Define the response closure
    // let response_closure = |response: String| async move {
    //     println!("📤 Response received: {:#}", response);
    // };

    // // Call the run method
    // if let Err(e) = client.run(request_stream, response_closure).await {
    //     eprintln!("❌ Error running client.run: {}", e);
    // }

    // println!("Starting extraction and hashing for model: {}", model_name);

    // let model_extractor=ModelExtractor::new(&model_name)

    // match models::ModelExtractor::new(model_name) {
    //     Ok(extractor) => {
    //         match extractor.extract_model() {
    //             Ok(_) => println!("✅ Extraction and hashing completed successfully."),
    //             Err(e) => eprintln!("❌ Extraction failed: {}", e),
    //         }
    //     }
    //     Err(e) => eprintln!("❌ Failed to initialize ModelExtractor: {}", e),
    // }

    // verify_model_blob(model_name);

    // Ok(())
    // const TRITON_URL: &str ="http://localhost:8000/v2";

    // let client = TritonClient::new(TRITON_URL.to_string());
    // match client.list_models().await {
    //     Ok(result) => println!("✅ Model listing complete. {:?}",result),
    //     Err(e) => println!("❌ Failed to list models: {:?}", e),
    // }

    // Create dummy input data
    // let mut input_data = HashMap::new();
    // let input_name = "data_0";
    // let input_shape = vec![3, 224, 224]; // As per the metadata you shared
    // let num_elements: usize = input_shape.iter().product();
    // let data = vec![0.5_f32; num_elements]; // Dummy data

    // // // Wrap inside TensorData and insert into HashMap
    // input_data.insert(input_name, (TensorData::F32(data), input_shape));
    // let mut input_data = HashMap::new();
    // input_data.insert("data_0", (TensorData::F32(vec![0.1; 3 * 224 * 224]), vec![3, 224, 224]));
    // // Run Inference with model extraction
    // client.run_inference(
    //     "densenet_onnx",
    //     input_data,
    // ).await.unwrap();
    // let mut raw_inputs = HashMap::new();
    // raw_inputs.insert("data_0".to_string(), TensorData::F32(vec![0.1; 3 * 224 * 224]));

    // let result = client.run_inference("densenet_onnx", raw_inputs).await?;

    // println!("Inference result: {:?}", result);

    // Ok(())
}
