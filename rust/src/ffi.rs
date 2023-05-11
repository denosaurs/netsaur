use safetensors::{serialize, SafeTensors};
use std::slice::{from_raw_parts, from_raw_parts_mut};

use crate::{
    decode_array, decode_json, length, CPUBackend, Dataset, PredictOptions, TrainOptions, RESOURCES,
};

#[no_mangle]
pub extern "C" fn ffi_backend_create(ptr: *const u8, len: usize, size_ptr: *mut u8) -> usize {
    let config = decode_json(ptr, len);
    let net_backend = CPUBackend::new(config);
    let mut buf: Vec<u8> = net_backend.size.iter().map(|x| *x as u8).collect();
    buf.insert(0, buf.len() as u8);
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
        backend[id].train(datasets, options.epochs, options.rate)
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
        let res = backend[id].predict(inputs);
        outputs.copy_from_slice(res.as_slice().unwrap());
    });
}

#[no_mangle]
// TODO: change this
#[allow(improper_ctypes_definitions)]
pub extern "C" fn ffi_backend_save(_id: usize) -> Vec<u8> {
    // temporary data
    let serialized = b"8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}\x00\x00\x00\x00";
    let loaded = SafeTensors::deserialize(serialized).unwrap();

    serialize(
        loaded
            .tensors()
            .iter()
            .map(|(name, view)| (name.to_string(), view)),
        &None,
    )
    .unwrap()
}
