#![allow(unsafe_op_in_unsafe_fn)]
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use std::ffi::{c_char, c_void, CStr, CString};
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

/// Number of prompt tokens decoded per call. Long prompts are fed in chunks of this size
/// so a large context does not need an equally large batch buffer.
const PROMPT_CHUNK: usize = 512;

/// Shortcut to convert Rust errors to Python errors.
fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

///
///
/// 
/// 
/// 
#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> String{
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();

        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    return out;
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.pending[..valid]));
                    match err.error_len() {
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        Some (bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    fn finish(&mut self) -> String {
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        rest
    }
}


fn render_chat(template: &CStr, messages: &[(String, String)], add_assistant: bool) -> Result<String, String> {
    let owned: Vec<(CString, CString)> = messages
        .iter()
        .map(|(role, content)| Ok((CString::new(role.as_str())?, CString::new(content.as_str())?)))
        .collect::<Result<_, std::ffi::NulError>>()
        .map_err(|e| format!("message contains a NUL byte: {e}"))?;

    let chat: Vec<llama_cpp_sys_2::llama_chat_message> = owned
        .iter()
        .map(|(role, content)| llama_cpp_sys_2::llama_chat_message {
        role: role.as_ptr(),
        content: content.as_ptr(),
    })
    .collect();

    let total: usize = messages.iter().map(|(r, c)| r.len() + c.len()).sum();

    let mut buf: Vec<u8> = vec![0; total * 2 + 256];
    let apply = |buf: &mut Vec<u8>| -> i32 {
        let length = i32::try_from(buf.len()).unwrap_or(i32::MAX);
        unsafe {
            llama_cpp_sys_2::llama_chat_apply_template(
                template.as_ptr(), 
                chat.as_ptr(), 
                chat.len(), 
                add_assistant, 
                buf.as_mut_ptr().cast::<c_char>(), 
                length
            )
        }
    };

    let mut res = apply(&mut buf);
    if res < 0 {
        return Err("llama.cpp does not suppor this chat template".to_string());
    }
    if res as usize > buf.len() {
        buf.resize(res as usize, 0);
        res = apply(&mut buf);
        if res < 0 {
            return Err("llama.cpp does not support this chat template".to_string());
        }
    }

    buf.truncate(res as usize);
    String::from_utf8(buf).map_err(|e| e.to_string())
}

/// Manages lifecycle of quantized models
#[pyclass]
pub struct ModelManager {
    backend: LlamaBackend,
    model: Option<LlamaModel>,
    current_name: String,
    last_usage: (usize, usize),
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
                last_usage: (0, 0),
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
        if self.current_name == name && self.model.is_some() {
            return Ok(format!("{} is already loaded.", name));
        }

        // Clear existing model
        self.model = None;
        self.current_name.clear();

        let model_params: LlamaModelParams = LlamaModelParams::default();
        let path: &Path = Path::new(&gguf_path);
        if !path.exists() {
            return Err(PyValueError::new_err(format!("Model file not found: {gguf_path}")));
        }
        // Load Quantized Model
        // This automatically handles the "shape mismatch" by reading the GGUF metadata
        let model: LlamaModel = LlamaModel::load_from_file(&self.backend, path, &model_params).map_err(to_py_err)?;

        self.model = Some(model);
        self.current_name = name;

        Ok(format!("Successfully loaded: {}", self.current_name))
    }

    pub fn current_model(&self) -> String {
        self.current_name.clone()
    }

    /// Returns the exact token count for a given string using the model's tokenizer.
    pub fn count_tokens(&self, text: String) -> PyResult<usize> {
        let model: &LlamaModel = self.model.as_ref().ok_or_else(|| PyRuntimeError::new_err("No model loaded"))?;

        let tokens = model
            .str_to_token(&text, AddBos::Never)
            .map_err(to_py_err)?;

        Ok(tokens.len())
    }

    pub fn n_ctx_train(&self) -> PyResult<u32> {
        let model = self.model.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("No model loaded"))?;
        Ok(model.n_ctx_train())
    }


    #[pyo3(signature = (messages, add_assistant = true, template = None))]
    pub fn apply_chat_template(
        &self,
        messages: Vec<(String, String)>,
        add_assistant: bool,
        template: Option<String>,
    ) -> PyResult<String> {
        let model = self.model.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("No model loaded"))?;
        let template_c: CString = match template {
            Some(name) => CString::new(name).map_err(to_py_err)?,

            None => model.chat_template(None).map_err(to_py_err)?.as_c_str().to_owned(),

        };
        render_chat(&template_c, &messages, add_assistant).map_err(PyRuntimeError::new_err)
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
    #[pyo3(signature = (
        prompt,
        callback,
        temperature = 0.7,
        max_tokens = 1024, 
        ctx_size = 8192,
        top_k = 40,
        top_p = 0.95,
        repeat_penalty = 1.0,
        seed = None,
        add_bos = true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self, 
        py: Python<'_>,
        prompt: String, 
        callback: PyObject, 
        temperature: f32,
        max_tokens: usize,
        ctx_size: u32,
        top_k: i32,            
        top_p: f32,            
        repeat_penalty: f32,
        seed: Option<u32>,
        add_bos: bool,
    ) -> PyResult<String> {
        let model: &LlamaModel = self.model.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("No model loaded"))?;
        
        // Context configureation
        if ctx_size < 64 {
            return Err(PyValueError::new_err("ctx_size must be at least 64"));
        }
        // let ctx_size: u32 = 8192;
        let ctx_params: LlamaContextParams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_size)) // Total context window
            .with_n_batch(PROMPT_CHUNK.min(ctx_size as usize) as u32); // Tokens processed per decode call
        
        let mut ctx: llama_cpp_2::context::LlamaContext<'_> = model.new_context(&self.backend, ctx_params).map_err(to_py_err)?;
        
        // Random seed for sampling
        let seed: u32 = seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u32)
                .unwrap_or(0)
        });

        let mut stages: Vec<LlamaSampler> = Vec::new();
        if repeat_penalty != 1.0 {
            stages.push(LlamaSampler::penalties(64, repeat_penalty, 0.0, 0.0));
        }

        if temperature <= 0.0 {
            stages.push(LlamaSampler::greedy());
        } else {
            stages.push(LlamaSampler::temp(temperature));
            stages.push(LlamaSampler::top_k(top_k));
            stages.push(LlamaSampler::top_p(top_p, 1));
            stages.push(LlamaSampler::dist(seed));
        }

        // Sampler configuration
        let mut sampler: LlamaSampler = LlamaSampler::chain_simple(stages);

        // Tokenize prompt
        let bos = if add_bos {AddBos::Always} else {AddBos::Never};
        let tokens: Vec<llama_cpp_2::token::LlamaToken> = model
            .str_to_token(&prompt, bos)
            .map_err(to_py_err)?;
        
        if tokens.is_empty() {
            return Err(PyValueError::new_err("empty prompt"));
        }
        if tokens.len() + 1 >= ctx_size as usize {
            self.last_usage = (tokens.len(), 0);
            return Ok("context_full".to_string());
        }
        // Create batch for processing
        let mut batch: LlamaBatch<'_> = LlamaBatch::new(PROMPT_CHUNK, 1);
        let last = tokens.len() - 1;

        // Add all prompt tokens to batch
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == last).map_err(to_py_err)?;
            if batch.n_tokens() as usize == PROMPT_CHUNK || i == last {
                ctx.decode(&mut batch).map_err(to_py_err)?;
                if i != last {
                    batch.clear();
                }
            }
        }
        
        let mut n_cur: i32 = tokens.len() as i32;
        let room: usize = ctx_size as usize - tokens.len() - 1;
        let budget: usize = max_tokens.min(room);
        let mut produced: usize = 0;

        let mut stop_reason: &str = if max_tokens <= room {"length_limit"} else {"context_full"};

        let mut text = Utf8Stream::default();

                // Generation loop
                while produced < budget {
                    // Sample next token
                    let next_token: llama_cpp_2::token::LlamaToken= sampler.sample(&ctx, batch.n_tokens() - 1);
                    
                    // Check for end of generation
                    if model.is_eog_token(next_token) {
                        stop_reason = "complete";
                        break;
                    }
                    produced += 1;

                    // Convert token to text
                    let output_bytes: Vec<u8> = model.token_to_bytes(next_token, Special::Plaintext).map_err(to_py_err)?;
                    let piece: String = text.push(&output_bytes);
                    
                    /*  
                    // [DEBUG]: 
                    println!("DEBUG: Generated token: [{}]", output_str); 
                    */

                    if !piece.is_empty() {
                        let answer: PyObject = callback.call1(py, (piece,))?;
                        if let Ok(false) = answer.extract::<bool>(py){
                            stop_reason = "stopped";
                            break;
                        }
                    }
                


                    // Prepare next decode
                batch.clear();
                batch.add(next_token, n_cur, &[0], true).map_err(to_py_err)?;
                n_cur += 1;

                ctx.decode(&mut batch).map_err(to_py_err)?;
            }

            let rest: String = text.finish();
            if !rest.is_empty() {
                callback.call1(py, (rest,))?;
            }
            self.last_usage = (tokens.len(), produced);
            Ok(stop_reason.to_string())
    }
   
    pub fn last_usage(&self) -> (usize, usize) {
        self.last_usage
            
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

#[cfg(test)] 
mod tests {
    use std::assert_eq;

use super::Utf8Stream;

    #[test]
    fn multibyte_characters_split_across_tokens_survive() {
        let mut s = Utf8Stream::default();
        let arrow = "→".as_bytes();
        let mut out = s.push(&arrow[..1]);
        out += &s.push(&arrow[1..2]);
        out += &s.push(&arrow[2..]);
        out += &s.push(" ℕ x".as_bytes());
        out += &s.finish();
        assert_eq!(out, "→ ℕ x");
    }

    #[test]
    fn invalid_bytes_become_replacement_characters() {
        let mut s = Utf8Stream::default();
        let out = s.push(&[b'a', 0xff, b'b']);
        assert_eq!(out, "a\u{FFFD}b");
    }
}