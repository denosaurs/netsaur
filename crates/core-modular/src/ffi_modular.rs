use crate::{
    decode_json, ActivationCPULayer, BackendConfig, BatchNorm1DCPULayer, BatchNorm2DCPULayer,
    BatchNormTensors, CPUCost, CPULayer, CPUOptimizer, CPUPostProcessor, CPUScheduler,
    Conv2DCPULayer, ConvTensors, ConvTranspose2DCPULayer, Cost, Dataset, DenseCPULayer,
    DenseTensors, Dropout1DCPULayer, Dropout2DCPULayer, EmbeddingCPULayer, FlattenCPULayer,
    GetTensor, LSTMCPULayer, Layer, Logger, Pool2DCPULayer, PostProcessor, SoftmaxCPULayer, Tensor,
    Tensors, Timer,
};

use ndarray::{ArrayD, ArrayViewD, AssignElem, Dimension, IxDyn};
use safetensors::{serialize, SafeTensors};
use serde::Deserialize;
use std::boxed::Box;
use std::mem::transmute;
use std::ptr::read;
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// Compare performance with non-modular backend

#[no_mangle]
pub unsafe extern "C" fn ffi_layer_create(
    ptr: *const u8,
    len: usize,
    input_size_ptr: *const u64,
    input_size_len: usize,
) -> usize {
    let config = decode_json(ptr, len);
    let input_size = from_raw_parts(input_size_ptr, input_size_len);
    let layer = create_layer(
        config,
        input_size.iter().map(|x| *x as usize).collect(),
        None,
    );
    transmute::<Box<CPULayer>, usize>(Box::new(layer))
}

#[no_mangle]
pub unsafe extern "C" fn ffi_layer_forward(
    layer: *mut CPULayer,
    data_ptr: *const f32,
    shape_ptr: *const u64,
    shape_len: usize,
    output_ptr: *mut f32,
    training: bool,
) {
    let mut layer = read(layer);
    let input_shape = IxDyn(
        std::slice::from_raw_parts(shape_ptr, shape_len)
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>()
            .as_slice(),
    );
    let raw_inputs = from_raw_parts(data_ptr, input_shape.size());
    let inputs: ArrayD<f32> = ArrayD::from_shape_vec(input_shape, raw_inputs.to_vec()).unwrap();
    let outputs = layer.forward_propagate(inputs, training);
    println!("HA {:?}", &outputs);
    let output_arr = from_raw_parts_mut(output_ptr, outputs.raw_dim().size());
    println!("HA");
    output_arr.copy_from_slice(outputs.as_slice().unwrap());
}

#[no_mangle]
pub unsafe extern "C" fn ffi_layer_backward(
    layer: *mut CPULayer,
    data_ptr: *const f32,
    shape_ptr: *const u64,
    shape_len: usize,
    output_ptr: *mut f32,
) {
    let mut layer = read(layer);
    let input_shape = IxDyn(
        std::slice::from_raw_parts(shape_ptr, shape_len)
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>()
            .as_slice(),
    );
    let raw_inputs = from_raw_parts(data_ptr, input_shape.size());

    let inputs: ArrayD<f32> = ArrayD::from_shape_vec(input_shape, raw_inputs.to_vec()).unwrap();

    let outputs = layer.backward_propagate(inputs);

    let output_arr = from_raw_parts_mut(output_ptr, outputs.raw_dim().size());

    output_arr.copy_from_slice(outputs.as_slice().unwrap());
}

#[no_mangle]
pub unsafe extern "C" fn ffi_optimize(
    optimizer: *mut CPUOptimizer,
    learning_rate: f32,
    epoch: usize,
    layers_ptr: *const u64,
    layers_len: usize,
) {
    let mut optimizer = read(optimizer);
    let layers = from_raw_parts(layers_ptr, layers_len);

    let mut actual_layers: Vec<CPULayer> = Vec::new();

    layers
        .to_vec()
        .iter()
        .for_each(|x| actual_layers.push(read(*x as *mut CPULayer)));

    optimizer.update_grads(&mut actual_layers, learning_rate, epoch);
}

#[no_mangle]
pub unsafe extern "C" fn ffi_optimizer_create(
    ptr: *const u8,
    len: usize,
    layers_ptr: *const u64,
    layers_len: usize,
) -> usize {
    let config = decode_json(ptr, len);

    let layers = from_raw_parts(layers_ptr, layers_len);

    let mut actual_layers: Vec<CPULayer> = Vec::new();

    layers
        .to_vec()
        .iter()
        .for_each(|x| actual_layers.push(read(*x as *mut CPULayer)));

    let optimizer = CPUOptimizer::from(config, &mut actual_layers);
    transmute::<Box<CPUOptimizer>, usize>(Box::new(optimizer))
}

#[derive(Deserialize)]
struct CostConfig {
    pub cost: Cost,
}

#[no_mangle]
pub unsafe extern "C" fn ffi_cost_create(ptr: *const u8, len: usize) -> usize {
    let config: CostConfig = decode_json(ptr, len);
    let cost = CPUCost::from(config.cost);
    transmute::<Box<CPUCost>, usize>(Box::new(cost))
}

#[no_mangle]
pub unsafe extern "C" fn ffi_cost_d(
    cost: *const CPUCost,
    shape_ptr: *const u64,
    shape_len: usize,
    y_ptr: *const f32,
    y_hat_ptr: *const f32,
    output_ptr: *mut f32,
) {
    let cost = read(cost);
    let shape = IxDyn(
        std::slice::from_raw_parts(shape_ptr, shape_len)
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>()
            .as_slice(),
    );
    let raw_y = from_raw_parts(y_ptr, shape.size());
    let raw_y_hat = from_raw_parts(y_hat_ptr, shape.size());

    let y = ArrayD::from_shape_vec(shape, raw_y.to_vec()).unwrap();
    let y_hat = ArrayD::from_shape_vec(y.raw_dim(), raw_y_hat.to_vec()).unwrap();

    let loss = (cost.prime)(y_hat.view(), y.view());

    let output_arr = from_raw_parts_mut(output_ptr, loss.raw_dim().size());

    output_arr.copy_from_slice(loss.as_slice().unwrap());
}

#[no_mangle]
pub unsafe extern "C" fn ffi_cost(
    cost: *const CPUCost,
    shape_ptr: *const u64,
    shape_len: usize,
    y_ptr: *const f32,
    y_hat_ptr: *const f32,
) -> f32 {
    let cost = read(cost);
    println!("HER");
    let shape = IxDyn(
        std::slice::from_raw_parts(shape_ptr, shape_len)
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>()
            .as_slice(),
    );
    println!("HER");
    let raw_y = std::slice::from_raw_parts(y_ptr, shape.size());
    println!("HER");
    let raw_y_hat = std::slice::from_raw_parts(y_hat_ptr, shape.size());
    println!("HER");
    let y = ArrayD::from_shape_vec(shape, raw_y.to_vec()).unwrap();
    let y_hat = ArrayD::from_shape_vec(y.raw_dim(), raw_y_hat.to_vec()).unwrap();
    println!("\n\n{:?} \n\n{:?}", &y, &y_hat);
    println!("HER");
    (cost.cost)(y_hat.view(), y.view())
}

pub fn create_layer(layer: Layer, size: Vec<usize>, tensors: Option<Tensors>) -> CPULayer {
    match layer.clone() {
        Layer::Activation(config) => {
            let layer = ActivationCPULayer::new(config, IxDyn(&size));
            CPULayer::Activation(layer)
        }
        Layer::Conv2D(config) => {
            let layer = Conv2DCPULayer::new(config, IxDyn(&size), tensors);
            CPULayer::Conv2D(layer)
        }
        Layer::ConvTranspose2D(config) => {
            let layer = ConvTranspose2DCPULayer::new(config, IxDyn(&size), tensors);
            CPULayer::ConvTranspose2D(layer)
        }
        Layer::BatchNorm1D(config) => {
            let layer = BatchNorm1DCPULayer::new(config, IxDyn(&size), tensors);
            CPULayer::BatchNorm1D(layer)
        }
        Layer::BatchNorm2D(config) => {
            let layer = BatchNorm2DCPULayer::new(config, IxDyn(&size), tensors);
            CPULayer::BatchNorm2D(layer)
        }
        Layer::Dropout1D(config) => {
            let layer = Dropout1DCPULayer::new(config, IxDyn(&size));
            CPULayer::Dropout1D(layer)
        }
        Layer::Dropout2D(config) => {
            let layer = Dropout2DCPULayer::new(config, IxDyn(&size));
            CPULayer::Dropout2D(layer)
        }
        Layer::Dense(config) => {
            let layer = DenseCPULayer::new(config, IxDyn(&size), tensors);
            CPULayer::Dense(layer)
        }
        Layer::Embedding(config) => {
            let layer = EmbeddingCPULayer::new(config, IxDyn(&size));
            CPULayer::Embedding(layer)
        }
        Layer::Flatten => {
            let layer = FlattenCPULayer::new(IxDyn(&size));
            CPULayer::Flatten(layer)
        }
        Layer::LSTM(config) => {
            let layer = LSTMCPULayer::new(config, IxDyn(&size), tensors);
            CPULayer::LSTM(layer)
        }
        Layer::Pool2D(config) => {
            let layer = Pool2DCPULayer::new(config, IxDyn(&size));
            CPULayer::Pool2D(layer)
        }
        Layer::Softmax(config) => {
            let layer = SoftmaxCPULayer::new(config, IxDyn(&size));
            CPULayer::Softmax(layer)
        }
    }
}
