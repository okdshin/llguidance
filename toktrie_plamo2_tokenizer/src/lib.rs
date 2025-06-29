use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

pub struct PyPLaMo2Tokenizer {
    py_tokenizer: PyObject,
    
    // TokTrie integration
    tok_trie: TokTrie,
}

impl PyPLaMo2Tokenizer {
    /// Create tokenizer from an existing Python Plamo2Tokenizer object
    pub fn from_python_object(py_tokenizer: PyObject) -> Result<Self> {
        Python::with_gil(|py| {
            // Get vocab size
            let vocab_size: usize = py_tokenizer.getattr(py, "vocab_size")?.extract(py)?;
            
            // Extract token information for TokTrie
            let (info, token_bytes) = Self::extract_token_info(py, &py_tokenizer, vocab_size)?;
            let tok_trie = TokTrie::from(&info, &token_bytes);
            
            Ok(PyPLaMo2Tokenizer {
                py_tokenizer,
                tok_trie,
            })
        })
    }
    
    /// Create tokenizer from vocab file (alternative constructor)
    pub fn new(vocab_file: &str) -> Result<Self> {
        Python::with_gil(|py| {
            // Import the Python tokenizer module
            let tokenizer_module = py.import("transformers")?;
            
            // Create the tokenizer instance
            let tokenizer_class = tokenizer_module.getattr("Plamo2Tokenizer")?;
            let py_tokenizer = tokenizer_class.call1((vocab_file,))?;
            
            // Use the from_python_object method
            Self::from_python_object(py_tokenizer.into())
        })
    }
    
    fn extract_token_info(
        py: Python,
        py_tokenizer: &PyObject,
        vocab_size: usize,
    ) -> Result<(TokRxInfo, Vec<Vec<u8>>)> {
        // Get all tokens and their information
        let mut token_bytes = vec![Vec::new(); vocab_size];
        
        // Extract tokens by iterating through vocab
        for token_id in 0..vocab_size {
            let token_str: String = py_tokenizer
                .call_method1(py, "_convert_id_to_token", (token_id,))?
                .extract(py)?;
            
            if token_str.starts_with("<0x") && token_str.ends_with(">") && token_str.len() == 6 {
                // Byte token
                let hex_str = &token_str[3..5];
                if let Ok(byte_val) = u8::from_str_radix(hex_str, 16) {
                    token_bytes[token_id] = vec![byte_val];
                }
            } else if token_str.starts_with('<') && token_str.ends_with('>') {
                // Special token
                let mut spec_bytes = Vec::with_capacity(token_str.len() + 1);
                spec_bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
                spec_bytes.extend_from_slice(token_str.as_bytes());
                token_bytes[token_id] = spec_bytes;
            } else {
                // Regular token
                token_bytes[token_id] = token_str.as_bytes().to_vec();
            }
        }
        
        // Extract special token IDs
        let mut info = TokRxInfo::new(
            vocab_size as u32,
            Self::get_special_token_id(py, py_tokenizer, "eos_token_id").unwrap_or(0),
        );
        info.tok_bos = Self::get_special_token_id(py, py_tokenizer, "bos_token_id");
        info.tok_unk = Self::get_special_token_id(py, py_tokenizer, "unk_token_id");
        info.tok_pad = Self::get_special_token_id(py, py_tokenizer, "pad_token_id");
        info.tok_end_of_turn = None;
        
        Ok((info, token_bytes))
    }
    
    fn get_special_token_id(
        py: Python,
        py_tokenizer: &PyObject,
        attr_name: &str,
    ) -> Option<u32> {
        py_tokenizer
            .getattr(py, attr_name)
            .ok()?
            .extract::<u32>(py)
            .ok()
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Python::with_gil(|py| {
            let result = self.py_tokenizer.call_method1(py, "_tokenize", (text,))?;
            let token_list = result.downcast_bound::<PyList>(py)
                .map_err(|e| anyhow::anyhow!("Failed to downcast to PyList: {}", e))?;
            
            let mut token_ids = Vec::new();
            for token in token_list.iter() {
                let token_str: String = token.extract()
                    .map_err(|e| anyhow::anyhow!("Failed to extract token string: {}", e))?;
                let token_id: u32 = self.py_tokenizer
                    .call_method1(py, "_convert_token_to_id", (&token_str,))?
                    .extract(py)
                    .map_err(|e| anyhow::anyhow!("Failed to extract token ID: {}", e))?;
                token_ids.push(token_id);
            }
            
            Ok(token_ids)
        })
    }
    
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        Python::with_gil(|py| {
            let py_list = PyList::new(py, token_ids)?;
            let result = self.py_tokenizer.call_method1(py, "decode", (&py_list,))?;
            let decoded: String = result.extract(py)
                .map_err(|e| anyhow::anyhow!("Failed to extract decoded string: {}", e))?;
            Ok(decoded)
        })
    }
    
    pub fn to_env(self) -> TokEnv {
        Arc::new(PyPLaMo2TokenizerEnv { tokenizer: self })
    }
}

pub struct PyPLaMo2TokenizerEnv {
    tokenizer: PyPLaMo2Tokenizer,
}

impl TokenizerEnv for PyPLaMo2TokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.tokenizer.tok_trie
    }
    
    fn find_byte_token(&self, byte: u8) -> Option<TokenId> {
        Python::with_gil(|py| {
            let byte_token = format!("<0x{:02X}>", byte);
            self.tokenizer.py_tokenizer
                .call_method1(py, "_convert_token_to_id", (&byte_token,))
                .ok()?
                .extract::<u32>(py)
                .ok()
        })
    }
    
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        match String::from_utf8(s.to_vec()) {
            Ok(text) => self.tokenizer.encode(&text).unwrap_or_else(|e| {
                eprintln!("PLaMo2 tokenization error: {}", e);
                Vec::new()
            }),
            Err(_) => {
                // Handle invalid UTF-8 by tokenizing individual bytes
                let mut result = Vec::new();
                for &byte in s {
                    // Look for byte token in format <0xXX>
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Ok(token_ids) = self.tokenizer.encode(&byte_token) {
                        result.extend(token_ids);
                    } else {
                        // If byte token doesn't exist, try to find it in vocab
                        if let Some(token_id) = self.find_byte_token(byte) {
                            result.push(token_id);
                        }
                    }
                }
                result
            }
        }
    }
}

/// Public function to create LLTokenizer from Python Plamo2Tokenizer
pub fn tokenv_from_plamo2_tokenizer(py_tokenizer: PyObject) -> Result<TokEnv> {
    let tokenizer = PyPLaMo2Tokenizer::from_python_object(py_tokenizer)?;
    Ok(tokenizer.to_env())
}
