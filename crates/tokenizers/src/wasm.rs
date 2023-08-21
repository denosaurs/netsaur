use crate::RESOURCES;
use std::str::FromStr;
use tokenizers::tokenizer::Tokenizer;
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
