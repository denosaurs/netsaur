mod ffi;
mod gpu;
mod tensor;
mod types;
mod util;

pub use gpu::*;

pub use ffi::*;
pub use tensor::*;
pub use types::*;
pub use util::*;

use std::cell::RefCell;

pub struct Resources {
    pub backend: RefCell<Vec<Backend>>,
}

impl Resources {
    pub fn new() -> Self {
        Self {
            backend: RefCell::new(Vec::new()),
        }
    }
}

thread_local! {
    pub static RESOURCES: Resources = Resources::new();
}
