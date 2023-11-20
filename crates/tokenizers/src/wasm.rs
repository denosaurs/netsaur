use crate::RESOURCES;
use std::{collections::HashMap, str::FromStr};
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
pub fn wasm_tokenizer_save(id: usize, pretty: bool) -> String {
    let mut data: String = String::new();
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        let tokenizer = &tokenizers[id];
        data = tokenizer.to_string(pretty).unwrap();
    });
    data
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
pub fn wasm_tokenizer_encode(id: usize, string: String) -> Vec<u32> {
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

#[wasm_bindgen]
pub fn wasm_tokenizer_get_vocab(id: usize, with_added_tokens: bool) -> JsValue {
    let mut data: HashMap<String, u32> = HashMap::new();
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        data = tokenizers[id].get_vocab(with_added_tokens)
    });
    serde_wasm_bindgen::to_value(&data).unwrap()
}

#[wasm_bindgen]
pub fn wasm_tokenizer_get_vocab_size(id: usize, with_added_tokens: bool) -> usize {
    let mut data: usize = 0;
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        data = tokenizers[id].get_vocab_size(with_added_tokens)
    });
    data
}

#[wasm_bindgen]
pub fn wasm_tokenizer_decode(id: usize, ids: &[u32], skip_special_tokens: bool) -> String {
    let mut data: String = String::new();
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        data = tokenizers[id].decode(ids, skip_special_tokens).unwrap()
    });
    data
}

#[wasm_bindgen]
pub fn wasm_tokenizer_token_to_id(id: usize, token: String) -> u32 {
    let mut data: u32 = 0;
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        data = tokenizers[id].token_to_id(token.as_str()).unwrap()
    });
    data
}

#[wasm_bindgen]
pub fn wasm_tokenizer_id_to_token(id: usize, token_id: u32) -> String {
    let mut data: String = String::new();
    RESOURCES.with(|cell| {
        let tokenizers = cell.tokenizer.borrow_mut();
        data = tokenizers[id].id_to_token(token_id).unwrap()
    });
    data
}