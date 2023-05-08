mod cpu;
mod types;
mod util;
#[cfg(not(target_arch = "wasm32"))]
mod ffi;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use cpu::*;
pub use types::*;
pub use util::*;
#[cfg(not(target_arch = "wasm32"))]
pub use ffi::*;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

use std::cell::RefCell;

pub struct Resources {
    pub backend: RefCell<Option<CPUBackend>>,
}

impl Resources {
    pub fn new() -> Self {
        Self {
            backend: RefCell::new(None),
        }
    }
}

thread_local! {
    pub static RESOURCES: Resources = Resources::new();
}