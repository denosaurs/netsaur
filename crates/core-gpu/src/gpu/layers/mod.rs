mod activation;
mod dense;

pub use activation::*;
pub use dense::*;

use ndarray::ArrayD;

pub enum GPULayer {
    Activation(ActivationGPULayer),
    Dense(DenseGPULayer),
}

impl GPULayer {
    pub fn output_size(&mut self) -> Vec<usize> {
        match self {
            GPULayer::Activation(layer) => layer.output_size(),
            GPULayer::Dense(layer) => layer.output_size(),
        }
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>, _training: bool) -> ArrayD<f32> {
        match self {
            GPULayer::Activation(layer) => layer.forward_propagate(inputs),
            GPULayer::Dense(layer) => layer.forward_propagate(inputs),
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            GPULayer::Activation(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Dense(layer) => layer.backward_propagate(d_outputs),
        }
    }

    pub fn reset(&mut self, batches: usize) {
        match self {
            GPULayer::Activation(layer) => layer.reset(batches),
            GPULayer::Dense(layer) => layer.reset(batches),
        }
    }
}
