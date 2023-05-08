use ndarray::ArrayD;

use crate::{DenseCPULayer, ActivationCPULayer, Conv2DCPULayer, Pool2DCPULayer, FlattenCPULayer, Dropout1DCPULayer, Dropout2DCPULayer};

pub enum CPULayer {
    Activation(ActivationCPULayer),
    Conv2D(Conv2DCPULayer),
    Dense(DenseCPULayer),
    Dropout1D(Dropout1DCPULayer),
    Dropout2D(Dropout2DCPULayer),
    Flatten(FlattenCPULayer),
    Pool2D(Pool2DCPULayer),
}

impl CPULayer {
    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            CPULayer::Activation(layer) => layer.forward_propagate(inputs),
            CPULayer::Conv2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Dense(layer) => layer.forward_propagate(inputs),
            CPULayer::Dropout1D(layer) => layer.forward_propagate(inputs),
            CPULayer::Dropout2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Flatten(layer) => layer.forward_propagate(inputs),
            CPULayer::Pool2D(layer) => layer.forward_propagate(inputs),
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        match self {
            CPULayer::Activation(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Conv2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Dense(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Dropout1D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Dropout2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Flatten(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Pool2D(layer) => layer.backward_propagate(d_outputs, rate),
        }
    }

    pub fn reset(&mut self, batches: usize) {
        match self {
            CPULayer::Activation(layer) => layer.reset(batches),
            CPULayer::Conv2D(layer) => layer.reset(batches),
            CPULayer::Dense(layer) => layer.reset(batches),
            CPULayer::Dropout1D(layer) => layer.reset(batches),
            CPULayer::Dropout2D(layer) => layer.reset(batches),
            CPULayer::Flatten(layer) => layer.reset(batches),
            CPULayer::Pool2D(layer) => layer.reset(batches),
        }
    }
}