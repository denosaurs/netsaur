use std::slice::{from_raw_parts, from_raw_parts_mut};

use crate::{CPUBackend, decode_json, RESOURCES, TrainOptions, decode_array, Dataset, PredictOptions, length};

#[no_mangle]
pub extern "C" fn ffi_backend_create(ptr: *const u8, len: usize) {
    let config = decode_json(ptr, len);
    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        backend.replace(CPUBackend::new(config))
    });
}

#[no_mangle]
pub extern "C" fn ffi_backend_train(
    buffer_ptr: *const u64,
    buffer_len: usize,
    options_ptr: *const u8,
    options_len: usize,
) {
    let buffer = unsafe { from_raw_parts(buffer_ptr, buffer_len) };
    let options: TrainOptions = decode_json(options_ptr, options_len);

    let mut datasets = Vec::new();
    for i in 0..options.datasets {
        let input = buffer[i * 2];
        let output = buffer[i * 2 + 1];
        datasets.push(Dataset {
            inputs: decode_array(input as *const f32, options.input_shape.clone()),
            outputs: decode_array(output as *const f32, options.output_shape.clone()),
        });
    }

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        backend
            .as_mut()
            .unwrap()
            .train(datasets, options.epochs, options.rate)
    });
}

#[no_mangle]
pub extern "C" fn ffi_backend_predict(
    buffer_ptr: *const f32,
    options_ptr: *const u8,
    options_len: usize,
    output_ptr: *mut f32,
) {
    let options: PredictOptions = decode_json(options_ptr, options_len);
    let inputs = decode_array(buffer_ptr, options.input_shape);

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        let res = backend.as_mut().unwrap().predict(inputs);
        let outputs = unsafe { from_raw_parts_mut(output_ptr, length(options.output_shape)) };
        outputs.copy_from_slice(res.as_slice().unwrap());
    });
}
