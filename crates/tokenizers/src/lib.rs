mod util;
mod wasm;
use tokenizers::{ModelWrapper, Tokenizer};

pub use util::*;

pub use wasm::*;

use std::cell::RefCell;

pub struct Resources {
    pub tokenizer: RefCell<Vec<Tokenizer>>,
    pub model: RefCell<Vec<ModelWrapper>>,
}

impl Resources {
    pub fn new() -> Self {
        Self {
            tokenizer: RefCell::new(Vec::new()),
            model: RefCell::new(Vec::new()),
        }
    }
}

thread_local! {
    pub static RESOURCES: Resources = Resources::new();
}
