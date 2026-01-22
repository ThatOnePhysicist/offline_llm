#![allow(unsafe_op_in_unsafe_fn)]
use pyo3::{ToPyErr, prelude::*};
use pyo3::exceptions::PyRuntimeError;
use candle_core::{Device, Tensor};
use candle_core::IndexOp;
use candle_transformers::models::quantized_qwen2::ModelWeights as QModel;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::path::Path;
/*
    This Rust module handles the backend of the offline chat with Python handling the front-end. 
    Specifically, Rust handles the model switching, device management (metal/cpu), and token streams. 
*/

/// Shortcut to convert Rust errors to Python errors.
fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Forward pass checking input tensor is on the correct device 
/// before running model.
fn run_hybrid_forward(
    model: &mut QModel,
    input: &Tensor, 
    start_pos: usize,
    device: &Device
) -> PyResult<Tensor>{
    // let device_input = input.to_device(device).map_err(to_py_err)?;
    model.forward(input, start_pos).map_err(to_py_err)
}

/// Manages lifecycle of quantized models
#[pyclass]
pub struct ModelManager {
    device: Device,
    current_model: Option<QModel>,
    tokenizer: Option<Tokenizer>,
    current_name: String,
}


#[pymethods]
impl ModelManager {
    /// Default attempts to initialize Metal and falls back to CPU. 
    #[new]
    pub fn new() -> PyResult<Self> {
        // let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let device = match Device::new_metal(0) {
            Ok(d) => {
                println!("Metal GPU initialized");
                d
            },
            Err(e) => {
                println!("Metal failed {:?}. Falling back to CPU.", e);
                Device::Cpu
            }
        };
        Ok(Self {
            device,
            current_model: None,
            tokenizer: None,
            current_name: String::new(),
        })
    }
    /// Loads model and tokenizer into memory
    /// 
    /// ### Arguments 
    /// * `name` - Model identifier. 
    /// * `gguf_path` - Absolute path to .gguf file. 
    /// 
    /// ### Note
    /// The tokenizer location is assumed to be in the same directory as the GGUF file
    /// with name `tokenizer.json`.
    pub fn load_model(&mut self, name: String, gguf_path: String) -> PyResult<String> {
        if self.current_name == name {
            return Ok(format!("{} is already loaded.", name));
        }

        // Clear existing model
        self.current_model = None;
        let path = Path::new(&gguf_path);
        
        // Open GGUF file
        let mut file = std::fs::File::open(path).map_err(to_py_err)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(to_py_err)?;

        // Load Quantized Model
        // This automatically handles the "shape mismatch" by reading the GGUF metadata
        let model = QModel::from_gguf(content, &mut file, &self.device).map_err(to_py_err)?;

        // Load Tokenizer (Usually stays as a separate json file in the GGUF folder)
        let tokenizer_path = path.parent().unwrap().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(to_py_err)?;

        self.current_model = Some(model);
        self.tokenizer = Some(tokenizer);
        self.current_name = name;

        Ok(format!("Successfully loaded GGUF model: {}", self.current_name))
    }

    /// Generates response and streams to Python
    /// 
    /// ### Arguments
    /// * `prompt` - Input prompt for model.
    /// * `callback` - The `stream_callback` function from `chat_script_r.py`.
    /// 
    /// ### Note
    /// This method uses a sampling temperature of 0.7 and stops generation if 
    /// an EOS token is produced. 
    pub fn generate(&mut self, prompt: String, callback: PyObject) -> PyResult<()> {
        let max_tokens = 2048;
        let model = self.current_model.as_mut().ok_or_else(|| PyRuntimeError::new_err("No model loaded"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| PyRuntimeError::new_err("No tokenizer loaded"))?;

        // Prompt encoding
        let tokens_obj = tokenizer.encode(prompt, true).map_err(to_py_err)?;
        let prompt_tokens = tokens_obj.get_ids().to_vec();
        let mut tokens = prompt_tokens.clone();
        // let prompt_len = prompt_tokens.len(); Maybe remove

        let mut logits_processor = LogitsProcessor::new(42, Some(0.7), None);
        // let mut prev_index = 0; MAYBE REMOVE
        
        Python::with_gil(|py|{
            for i in 0..max_tokens {

                // Release GIL 
                // Wrap the forward pass in allow_threads so the Python UI remains responsive
                // and the OS scheduler can handle the GPU driver efficiently.
                let next_token = py.allow_threads(|| -> PyResult<u32> {
                    let (input, start_pos) = if i==0 {
                        // Process whole prompt
                        let input = Tensor::new(&tokens[..], &self.device)
                            .map_err(to_py_err)?
                            .unsqueeze(0)
                            .map_err(to_py_err)?;
                    (input, 0)
                    } else {
                        // Process only last token
                        // create a tensor directly on target device (Metal)
                        let last_token = *tokens.last().unwrap();
                        let input = Tensor::new(&[last_token], &self.device)
                            .map_err(to_py_err)?
                            .unsqueeze(0)
                            .map_err(to_py_err)?;
                        (input, tokens.len()-1)
                    };
                    let logits = model.forward(&input, start_pos).map_err(to_py_err)?;
                    let logits = logits.squeeze(0).map_err(to_py_err)?;
                    let logits = logits.to_device(&Device::Cpu).map_err(to_py_err)?.to_dtype(candle_core::DType::F32).map_err(to_py_err)?;

                    let next_token = logits_processor.sample(&logits).map_err(to_py_err)?;
                    Ok(next_token)
                })?;
                tokens.push(next_token);
                // Stop tokens for Qwen2.
                if next_token == 151643 || next_token == 151645 {
                    break;
                }

                let new_text = tokenizer.decode(&[next_token], true).map_err(to_py_err)?;
                if !new_text.is_empty() {
                    callback.call1(py, (new_text,))?;
                }
            }
            Ok(())
        })
    }
}

/// Python import
/// 
/// This shows as `import model_handler` in the `chat_script_r.py` file.
#[pymodule]
fn model_handler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ModelManager>()?;
    Ok(())
}