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
        let backend = LlamaBackend::init().map_err(to_py_err)?;

        unsafe {
            llama_cpp_sys_2::llama_log_set(Some(silent_log_callback), std::ptr::null_mut());
        }

        Ok(Self {
            backend,
            model: None,
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
        let model_params = LlamaModelParams::default();
        let path = Path::new(&gguf_path);
        
        // Load Quantized Model
        // This automatically handles the "shape mismatch" by reading the GGUF metadata
        let model = LlamaModel::load_from_file(&self.backend, path, &model_params).map_err(to_py_err)?;

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
        let model = self.model.as_ref().ok_or_else(|| PyRuntimeError::new_err("No model loaded"))?;
        
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048));
        let mut ctx = model.new_context(&self.backend, ctx_params).map_err(to_py_err)?;
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
        let mut sampler = LlamaSampler::chain_simple([
            // LlamaSampler::repen(model.n_vocab(), 1.1, 64, 64),
            LlamaSampler::temp(0.7),
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            // LlamaSampler::greedy(),
            LlamaSampler::dist(seed),
        ]);

        // Prompt encoding
        // let tokens = model.str_to_token(&prompt, llama_cpp_2::model::AddBos::Always).map_err(to_py_err)?;
        let tokens = model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(to_py_err)?;

        let mut batch = LlamaBatch::new(2048, 1);
        
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == tokens.len() - 1);
        }
        
        // n_cur = 0;
        let mut n_cur = tokens.len() as i32;

        Python::with_gil(|py| {
                ctx.decode(&mut batch).map_err(to_py_err)?;

                for _ in 0..1024 {
                    let next_token= sampler.sample(&ctx, batch.n_tokens() - 1);

                    if model.is_eog_token(next_token) {
                        break;
                    }

                    let output_bytes = model.token_to_bytes(next_token, llama_cpp_2::model::Special::Plaintext).map_err(to_py_err)?;
                    let output_str = String::from_utf8_lossy(&output_bytes).into_owned();
                    
                    // Add this temporary debug line:
                    // println!("DEBUG: Generated token: [{}]", output_str);

                    callback.call1(py, (output_str,))?;
                    batch.clear();
                    batch.add(next_token, n_cur, &[0], true);
                    n_cur += 1;

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