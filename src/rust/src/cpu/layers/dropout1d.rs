use ndarray::{/*Array, Array2,*/ ArrayD /* , Axis, Ix2*/, IxDyn};
// use std::ops::{Add, AddAssign, Mul};

use crate::Dropout1DLayer;

pub struct Dropout1DCPULayer {}

impl Dropout1DCPULayer {
    pub fn new(_config: Dropout1DLayer, size: IxDyn) -> Self {
        let _input_size = [size[0], size[1]];

        Self {}
    }
    
    pub fn output_size(&self) -> Vec<usize> {
        vec![]
    }

    pub fn reset(&mut self, _batches: usize) {}

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        inputs
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        d_outputs
    }
}
