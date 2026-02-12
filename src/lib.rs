#![allow(unsafe_op_in_unsafe_fn)]
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use std::ffi::{c_char, c_void};
use std::time::{SystemTime, UNIX_EPOCH};

unsafe extern "C" fn silent_log_callback(
    _level: llama_cpp_sys_2::ggml_log_level, 
    _text: *const c_char,
    _user_data: *mut c_void,
) {
    // Intentionally left empty to supress output.
}
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
// fn run_hybrid_forward(
//     model: &mut QModel,
//     input: &Tensor, 
//     start_pos: usize,
//     device: &Device
// ) -> PyResult<Tensor>{
//     // let device_input = input.to_device(device).map_err(to_py_err)?;
//     model.forward(input, start_pos).map_err(to_py_err)
// }

/// Manages lifecycle of quantized models
#[pyclass]
pub struct ModelManager {
    backend: LlamaBackend,
    model: Option<LlamaModel>,
    current_name: String,
}


#[pymethods]
impl ModelManager {
    /// Default attempts to initialize Metal and falls back to CPU. 
    #[new]
    pub fn new() -> PyResult<Self> {
        // let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let backend: LlamaBackend = LlamaBackend::init().map_err(to_py_err)?;

        unsafe {
            llama_cpp_sys_2::llama_log_set(Some(silent_log_callback), std::ptr::null_mut());
        }

        Ok(
            Self {
                backend,
                model: None,
                current_name: String::new(),
            }
        )
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
        let model_params: LlamaModelParams = LlamaModelParams::default();
        let path: &Path = Path::new(&gguf_path);
        
        // Load Quantized Model
        // This automatically handles the "shape mismatch" by reading the GGUF metadata
        let model: LlamaModel = LlamaModel::load_from_file(&self.backend, path, &model_params).map_err(to_py_err)?;

        self.model = Some(model);
        self.current_name = name;

        Ok(format!("Successfully loaded: {}", self.current_name))
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
        let model: &LlamaModel = self.model.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("No model loaded"))?;
        
        // Context configureation
        let ctx_size: u32 = 4096;
        let ctx_params: LlamaContextParams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size)) // Total context window
            .with_n_batch(ctx_size); // Tokens processed per decode call
        
        let mut ctx: llama_cpp_2::context::LlamaContext<'_> = model.new_context(&self.backend, ctx_params).map_err(to_py_err)?;
        
        // Random seed for sampling
        let seed: u32 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        // Sampler configuration
        let mut sampler: LlamaSampler = LlamaSampler::chain_simple([
            // LlamaSampler::repen(model.n_vocab(), 1.1, 64, 64),
            LlamaSampler::temp(0.7),
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            // LlamaSampler::greedy(),
            LlamaSampler::dist(seed),
        ]);

        // Tokenize prompt
        let tokens: Vec<llama_cpp_2::token::LlamaToken> = model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(to_py_err)?;
        
        // Create batch for processing
        let mut batch: LlamaBatch<'_> = LlamaBatch::new(ctx_size as usize, 1);
        
        // Add all prompt tokens to batch
        for (i, &token) in tokens.iter().enumerate() {
            let _ = batch.add(token, i as i32, &[0], i == tokens.len() - 1);
        }
        
        let mut n_cur: i32 = tokens.len() as i32;

        Python::with_gil(|py: Python<'_>| {
                // Process prompt
                ctx.decode(&mut batch).map_err(to_py_err)?;

                // Calculate max tokens 
                let prompt_tokens: u32 = tokens.len() as u32;
                let max_new_tokens: usize = ((ctx_size.saturating_sub(prompt_tokens)).min(1024)) as usize;

                // Generation loop
                for _ in 0..max_new_tokens {
                    // Checking position before sampling for safety
                    if n_cur >= (ctx_size - 1) as i32 {
                        break;
                    }

                    // Sample next token
                    let next_token: llama_cpp_2::token::LlamaToken= sampler.sample(&ctx, batch.n_tokens() - 1);

                    // Check for end of generation
                    if model.is_eog_token(next_token) {
                        break;
                    }

                    // Convert token to text
                    let output_bytes: Vec<u8> = model.token_to_bytes(next_token, llama_cpp_2::model::Special::Plaintext).map_err(to_py_err)?;
                    let output_str: String = String::from_utf8_lossy(&output_bytes).into_owned();
                    
                    /*  
                    // [DEBUG]: 
                    println!("DEBUG: Generated token: [{}]", output_str); 
                    */

                    // Stream to Python
                    callback.call1(py, (output_str,))?;

                    // Prepare next decode
                    batch.clear();
                    let _ = batch.add(next_token, n_cur, &[0], true);
                    n_cur += 1;

                    // Check position before decoding for safety
                    if n_cur >= ctx_size as i32 {
                        break;
                    }

                    // Decode next position
                    ctx.decode(&mut batch).map_err(to_py_err)?;
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