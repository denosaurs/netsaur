use ndarray::ArrayD;

use crate::{DenseCPULayer};

pub enum CPULayer {
    Dense(DenseCPULayer),
}

impl CPULayer {
    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            CPULayer::Dense(layer) => layer.forward_propagate(inputs)
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        match self {
            CPULayer::Dense(layer) => layer.backward_propagate(d_outputs, rate)
        }
    }

    pub fn reset(&mut self, batches: usize) {
        match self {
            CPULayer::Dense(layer) => layer.reset(batches)
        }
    }
}