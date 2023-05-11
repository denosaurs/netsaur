use js_sys::Float32Array;
use ndarray::ArrayD;
use safetensors::{serialize, SafeTensors};

use wasm_bindgen::prelude::wasm_bindgen;

use crate::{CPUBackend, Dataset, PredictOptions, TrainOptions, RESOURCES};

#[wasm_bindgen]
pub fn wasm_backend_create(config: String) -> usize {
    let config = serde_json::from_str(&config).unwrap();
    let mut len = 0;
    let net_backend = CPUBackend::new(config);

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        len = backend.len();
        backend.push(net_backend);
    });
    len
}

#[wasm_bindgen]
pub fn wasm_backend_train(id: usize, buffers: Vec<Float32Array>, options: String) {
    let options: TrainOptions = serde_json::from_str(&options).unwrap();

    let mut datasets = Vec::new();
    for i in 0..options.datasets {
        let input = buffers[i * 2].to_vec();
        let output = buffers[i * 2 + 1].to_vec();
        datasets.push(Dataset {
            inputs: ArrayD::from_shape_vec(options.input_shape.clone(), input).unwrap(),
            outputs: ArrayD::from_shape_vec(options.output_shape.clone(), output).unwrap(),
        });
    }

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        backend[id].train(datasets, options.epochs, options.rate)
    });
}

#[wasm_bindgen]
pub fn wasm_backend_predict(id: usize, buffer: Float32Array, options: String) -> Float32Array {
    let options: PredictOptions = serde_json::from_str(&options).unwrap();
    let inputs = ArrayD::from_shape_vec(options.input_shape, buffer.to_vec()).unwrap();

    let mut res = ArrayD::zeros(options.output_shape);

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        res = backend[id].predict(inputs);
    });
    Float32Array::from(res.as_slice().unwrap())
}

#[wasm_bindgen]
pub fn wasm_backend_save(_id: usize) -> Vec<u8> {
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
