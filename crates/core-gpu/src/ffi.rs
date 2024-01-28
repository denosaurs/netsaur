use std::slice::{from_raw_parts, from_raw_parts_mut};

use crate::{
    decode_array, decode_json, length, Backend, Dataset, Logger, PredictOptions, TrainOptions,
    RESOURCES,
};

type AllocBufferFn = extern "C" fn(usize) -> *mut u8;

fn log(string: String) {
    println!("{}", string)
}

#[no_mangle]
pub extern "C" fn ffi_backend_create(ptr: *const u8, len: usize, alloc: AllocBufferFn) -> usize {
    let config = decode_json(ptr, len);
    let net_backend = Backend::new(config, Logger { log }, None);
    let buf: Vec<u8> = net_backend.size.iter().map(|x| *x as u8).collect();
    let size_ptr = alloc(buf.len());
    let output_shape = unsafe { from_raw_parts_mut(size_ptr, buf.len()) };
    output_shape.copy_from_slice(buf.as_slice());

    let mut len = 0;
    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        len = backend.len();
        backend.push(net_backend);
    });
    len
}

#[no_mangle]
pub extern "C" fn ffi_backend_train(
    id: usize,
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
        backend[id].train(datasets, options.epochs, options.batches, options.rate)
    });
}

#[no_mangle]
pub extern "C" fn ffi_backend_predict(
    id: usize,
    buffer_ptr: *const f32,
    options_ptr: *const u8,
    options_len: usize,
    output_ptr: *mut f32,
) {
    let options: PredictOptions = decode_json(options_ptr, options_len);
    let inputs = decode_array(buffer_ptr, options.input_shape);
    let outputs = unsafe { from_raw_parts_mut(output_ptr, length(options.output_shape)) };

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        let res = backend[id].predict(inputs, options.layers);
        outputs.copy_from_slice(res.as_slice().unwrap());
    });
}

#[no_mangle]
pub extern "C" fn ffi_backend_save(id: usize, alloc: AllocBufferFn) {
    RESOURCES.with(|cell| {
        let backend = cell.backend.borrow_mut();
        let data = backend[id].save();
        let file_ptr = alloc(data.len());
        let file = unsafe { from_raw_parts_mut(file_ptr, data.len()) };
        file.copy_from_slice(data.as_slice());
    });
}

#[no_mangle]
pub extern "C" fn ffi_backend_load(
    file_ptr: *const u8,
    file_len: usize,
    alloc: AllocBufferFn,
) -> usize {
    let buffer = unsafe { from_raw_parts(file_ptr, file_len) };
    let net_backend = Backend::load(buffer, Logger { log });
    let buf: Vec<u8> = net_backend.size.iter().map(|x| *x as u8).collect();
    let size_ptr = alloc(buf.len());
    let output_shape = unsafe { from_raw_parts_mut(size_ptr, buf.len()) };
    output_shape.copy_from_slice(buf.as_slice());

    let mut len = 0;
    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        len = backend.len();
        backend.push(net_backend);
    });
    len
}
