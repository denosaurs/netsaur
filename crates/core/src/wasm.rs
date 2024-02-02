use js_sys::{Array, Float32Array, Uint8Array};
use ndarray::ArrayD;

use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

use crate::{Backend, Dataset, Logger, PredictOptions, TrainOptions, RESOURCES};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

fn console_log(string: String) {
    log(string.as_str())
}

#[wasm_bindgen]
pub fn wasm_backend_create(config: String, shape: Array) -> usize {
    let config = serde_json::from_str(&config).unwrap();
    let mut len = 0;
    let logger = Logger { log: console_log };
    let net_backend = Backend::new(config, logger, None);
    shape.set_length(net_backend.size.len() as u32);
    for (i, s) in net_backend.size.iter().enumerate() {
        shape.set(i as u32, JsValue::from(*s))
    }

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
        backend[id].train(datasets, options.epochs, options.batches, options.rate)
    });
}

#[wasm_bindgen]
pub fn wasm_backend_predict(id: usize, buffer: Float32Array, options: String) -> Float32Array {
    let options: PredictOptions = serde_json::from_str(&options).unwrap();
    let inputs = ArrayD::from_shape_vec(options.input_shape, buffer.to_vec()).unwrap();

    let res = ArrayD::zeros(options.output_shape);

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        let _res = backend[id].predict(inputs, options.layers);
    });
    Float32Array::from(res.as_slice().unwrap())
}

#[wasm_bindgen]
pub fn wasm_backend_save(id: usize) -> Uint8Array {
    let mut buffer = Vec::new();
    RESOURCES.with(|cell| {
        let backend = cell.backend.borrow_mut();
        buffer = backend[id].save();
    });
    Uint8Array::from(buffer.as_slice())
}

#[wasm_bindgen]
pub fn wasm_backend_load(buffer: Uint8Array, shape: Array) -> usize {
    let mut len = 0;
    let logger = Logger { log: console_log };
    let net_backend = Backend::load(buffer.to_vec().as_slice(), logger);
    shape.set_length(net_backend.size.len() as u32);
    for (i, s) in net_backend.size.iter().enumerate() {
        shape.set(i as u32, JsValue::from(*s))
    }

    RESOURCES.with(|cell| {
        let mut backend = cell.backend.borrow_mut();
        len = backend.len();
        backend.push(net_backend);
    });
    len
}
