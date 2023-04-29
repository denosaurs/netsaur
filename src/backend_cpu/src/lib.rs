mod backend;
mod layers;
mod types;

pub use backend::*;
pub use layers::*;
pub use types::*;

use std::cell::RefCell;

pub struct Resources {
    backend: RefCell<Option<CPUBackend>>,
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

#[no_mangle]
pub extern "C" fn ops_backend_create(ptr: *const u8, len: usize) {
    let buffer = unsafe { std::slice::from_raw_parts(ptr, len) };
    let json = std::str::from_utf8(&buffer[0..len]).unwrap();
    let config = serde_json::from_str(&json).unwrap();
    println!("{:#?}", config);
    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        backend.replace(CPUBackend::new(config))
    });
}