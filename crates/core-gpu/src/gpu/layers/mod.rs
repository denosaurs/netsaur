mod activation;
mod dense;

pub use activation::*;
pub use dense::*;

use crate::{Tensors, WGPUBackend, WGPUBuffer};

pub enum GPULayer {
    Activation(ActivationGPULayer),
    Dense(DenseGPULayer),
}

impl GPULayer {
    pub fn outputs<'a>(&'a self) -> &'a WGPUBuffer {
        match self {
            GPULayer::Activation(layer) => &layer.outputs,
            GPULayer::Dense(layer) => &layer.outputs,
        }
    }

    pub fn d_inputs<'a>(&'a self) -> &'a WGPUBuffer {
        match self {
            GPULayer::Activation(layer) => &layer.d_inputs,
            GPULayer::Dense(layer) => &layer.d_inputs,
        }
    }

    pub fn forward_propagate(
        &self,
        backend: &mut WGPUBackend,
        inputs: &WGPUBuffer,
        _training: bool,
    ) {
        match self {
            GPULayer::Activation(layer) => layer.forward_propagate(backend, inputs),
            GPULayer::Dense(layer) => layer.forward_propagate(backend, inputs),
        }
    }

    pub fn backward_propagate(
        &self,
        backend: &mut WGPUBackend,
        inputs: &WGPUBuffer,
        d_outputs: &WGPUBuffer,
    ) {
        match self {
            GPULayer::Activation(layer) => layer.backward_propagate(backend, inputs, d_outputs),
            GPULayer::Dense(layer) => layer.backward_propagate(backend, inputs, d_outputs),
        }
    }

    pub fn reset(&mut self, backend: &mut WGPUBackend, batches: usize) {
        match self {
            GPULayer::Activation(layer) => layer.reset(backend, batches),
            GPULayer::Dense(layer) => layer.reset(backend, batches),
        }
    }

    pub fn save(&self, backend: &mut WGPUBackend) -> Tensors {
        match self {
            GPULayer::Activation(_) => Tensors::None,
            GPULayer::Dense(layer) => layer.save(backend),
        }
    }
}
