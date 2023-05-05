use ndarray::ArrayD;

use crate::{DenseCPULayer, ActivationCPULayer, Conv2DCPULayer, Pool2DCPULayer, FlattenCPULayer};

pub enum CPULayer {
    Dense(DenseCPULayer),
    Conv2D(Conv2DCPULayer),
    Pool2D(Pool2DCPULayer),
    Flatten(FlattenCPULayer),
    Activation(ActivationCPULayer),
}

impl CPULayer {
    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            CPULayer::Dense(layer) => layer.forward_propagate(inputs),
            CPULayer::Conv2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Pool2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Flatten(layer) => layer.forward_propagate(inputs),
            CPULayer::Activation(layer) => layer.forward_propagate(inputs)
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        match self {
            CPULayer::Dense(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Conv2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Pool2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Flatten(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Activation(layer) => layer.backward_propagate(d_outputs, rate)
        }
    }

    pub fn reset(&mut self, batches: usize) {
        match self {
            CPULayer::Dense(layer) => layer.reset(batches),
            CPULayer::Conv2D(layer) => layer.reset(batches),
            CPULayer::Pool2D(layer) => layer.reset(batches),
            CPULayer::Flatten(layer) => layer.reset(batches),
            CPULayer::Activation(layer) => layer.reset(batches)
        }
    }
}