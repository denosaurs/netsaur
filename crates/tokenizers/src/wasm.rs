use crate::RESOURCES;
use std::str::FromStr;
use tokenizers::{models::bpe::BPE, tokenizer::Tokenizer};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn wasm_tokenizer_from_json(json: String) -> usize {
    let tokenizer = Tokenizer::from_str(json.as_str()).unwrap().into();
    let mut len = 0;
    RESOURCES.with(|cell| {
        let mut tokenizers = cell.tokenizer.borrow_mut();
        len = tokenizers.len();
        tokenizers.push(tokenizer);
    });
    len
}

#[wasm_bindgen]
pub fn wasm_tokenizer_save(id: usize, pretty: bool) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        let tokenizer = &tokenizers[id];
        bytes = tokenizer.to_string(pretty).unwrap().as_bytes().to_vec();
    });
    bytes
}

#[wasm_bindgen]
pub fn wasm_bpe_default() -> usize {
    let model = BPE::default();
    let mut len = 0;
    RESOURCES.with(|cell| {
        let mut models = cell.model.borrow_mut();
        len = models.len();
        models.push(tokenizers::ModelWrapper::BPE(model));
    });
    len
}

#[wasm_bindgen]
pub fn wasm_tokenizer_tokenize(id: usize, string: String) -> Vec<u32> {
    let mut data: Vec<u32> = Vec::new();
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        data = tokenizers[id]
            .encode(string, false)
            .unwrap()
            .get_ids()
            .into_iter()
            .cloned()
            .collect()
    });
    data
}
