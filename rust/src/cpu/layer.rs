use ndarray::ArrayD;

use crate::{DenseCPULayer, BatchNorm2DCPULayer, ActivationCPULayer, Conv2DCPULayer, Pool2DCPULayer, FlattenCPULayer, Dropout1DCPULayer, Dropout2DCPULayer, SoftmaxCPULayer};

pub enum CPULayer {
    Activation(ActivationCPULayer),
    Conv2D(Conv2DCPULayer),
    Dense(DenseCPULayer),
    Dropout1D(Dropout1DCPULayer),
    Dropout2D(Dropout2DCPULayer),
    Flatten(FlattenCPULayer),
    Pool2D(Pool2DCPULayer),
    Softmax(SoftmaxCPULayer),
    BatchNorm2D(BatchNorm2DCPULayer),
}

impl CPULayer {
    pub fn output_size(&mut self) -> Vec<usize> {
        match self {
            CPULayer::Activation(layer) => layer.output_size(),
            CPULayer::BatchNorm2D(layer) => layer.output_size(),
            CPULayer::Conv2D(layer) => layer.output_size(),
            CPULayer::Dense(layer) => layer.output_size(),
            CPULayer::Dropout1D(layer) => layer.output_size(),
            CPULayer::Dropout2D(layer) => layer.output_size(),
            CPULayer::Flatten(layer) => layer.output_size(),
            CPULayer::Pool2D(layer) => layer.output_size(),
            CPULayer::Softmax(layer) => layer.output_size(),
        }
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>, training: bool) -> ArrayD<f32> {
        match self {
            CPULayer::Activation(layer) => layer.forward_propagate(inputs),
            CPULayer::BatchNorm2D(layer) => layer.forward_propagate(inputs, training),
            CPULayer::Conv2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Dense(layer) => layer.forward_propagate(inputs),
            CPULayer::Dropout1D(layer) => layer.forward_propagate(inputs),
            CPULayer::Dropout2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Flatten(layer) => layer.forward_propagate(inputs),
            CPULayer::Pool2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Softmax(layer) => layer.forward_propagate(inputs),
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        match self {
            CPULayer::Activation(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::BatchNorm2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Conv2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Dense(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Dropout1D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Dropout2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Flatten(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Pool2D(layer) => layer.backward_propagate(d_outputs, rate),
            CPULayer::Softmax(layer) => layer.backward_propagate(d_outputs, rate),
        }
    }

    pub fn reset(&mut self, batches: usize) {
        match self {
            CPULayer::Activation(layer) => layer.reset(batches),
            CPULayer::BatchNorm2D(layer) => layer.reset(batches),
            CPULayer::Conv2D(layer) => layer.reset(batches),
            CPULayer::Dense(layer) => layer.reset(batches),
            CPULayer::Dropout1D(layer) => layer.reset(batches),
            CPULayer::Dropout2D(layer) => layer.reset(batches),
            CPULayer::Flatten(layer) => layer.reset(batches),
            CPULayer::Pool2D(layer) => layer.reset(batches),
            CPULayer::Softmax(layer) => layer.reset(batches),
        }
    }
}