use std::path::{Path, PathBuf};
use tokenizers::{Encoding, Model, Tokenizer, TruncationDirection};

#[derive(Debug, thiserror::Error)]
pub enum TokError {
    #[error("Tokenizer file not found at {0}")]
    NotFound(PathBuf),
    #[error("Tokenization failed: {0}")]
    Fail(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub struct TextTokenizer {
    tok: Tokenizer,
    bos_id: Option<i64>,
    eos_id: Option<i64>,
}

impl TextTokenizer {
    /// Load tokenizer from model repo.
    /// Looks for: `<repo>/tokenizer.json` (preferred).
    pub fn from_repo(repo_dir: impl AsRef<Path>) -> Result<Self, TokError> {
        let repo = repo_dir.as_ref();
        let tj = repo.join("tokenizer.json");

        if !tj.exists() {
            return Err(TokError::NotFound(tj));
        }

        let tok = Tokenizer::from_file(&tj)
            .map_err(|e| TokError::Fail(format!("Tokenizer::from_file: {e}")))?;

        // best-effort special ids (if present)
        let model = tok.get_model();
        let bos_id = model.token_to_id("<s>").map(|id| id as i64);
        let eos_id = model.token_to_id("</s>").map(|id| id as i64);

        Ok(Self {
            tok,
            bos_id,
            eos_id,
        })
    }

    /// Encode a prompt to token ids. Adds BOS/EOS if requested.

    pub fn encode_ids(
        &self,
        prompt: &str,
        add_bos: bool,
        add_eos: bool,
        max_len: Option<usize>,
    ) -> Result<Vec<i64>, TokError> {
        let mut enc = self
            .tok
            .encode(prompt, true)
            .map_err(|e| TokError::Fail(format!("encode: {e}")))?;

        if let Some(m) = max_len {
            if enc.len() > m {
                enc.truncate(m, 0, TruncationDirection::Left);
            }
        }

        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();

        if add_bos {
            if let Some(bos) = self.bos_id {
                if ids.first().copied() != Some(bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if add_eos {
            if let Some(eos) = self.eos_id {
                if ids.last().copied() != Some(eos) {
                    ids.push(eos);
                }
            }
        }
        Ok(ids)
    }

    /// Decode token ids to text.
    pub fn decode_ids(&self, ids: &[i64]) -> Result<String, TokError> {
        let u32_ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.tok
            .decode(&u32_ids, true)
            .map_err(|e| TokError::Fail(format!("decode: {e}")))
    }

    pub fn bos_id(&self) -> Option<i64> {
        self.bos_id
    }
    pub fn eos_id(&self) -> Option<i64> {
        self.eos_id
    }
}

use crate::client::TensorData;
use std::collections::HashMap;

pub fn make_llm_inputs(token_ids: Vec<i64>) -> HashMap<String, (TensorData, Vec<usize>)> {
    let seq = token_ids.len();
    let bsz = 1usize;

    let mut map = HashMap::new();
    map.insert(
        "input_ids".to_string(),
        (TensorData::I64(token_ids), vec![bsz, seq]),
    );
    map.insert(
        "attention_mask".to_string(),
        (TensorData::I64(vec![1; seq]), vec![bsz, seq]),
    );
    map.insert(
        "position_ids".to_string(),
        (TensorData::I64((0..seq as i64).collect()), vec![bsz, seq]),
    );
    map
}
