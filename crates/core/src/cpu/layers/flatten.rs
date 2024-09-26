use ndarray::{ArrayD, Axis, Dimension, IxDyn};

use crate::FlattenLayer;

pub struct FlattenCPULayer {
    pub input_size: IxDyn,
    pub output_size: Vec<usize>,
}

impl FlattenCPULayer {
    pub fn new(config: FlattenLayer, size: IxDyn) -> Self {
        let output_size = IxDyn(&[size[0], size.size() / size[0]]);
        Self {
            input_size: size.clone(),
            output_size: vec![size[0], size.size() / size[0]],
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.output_size.clone()
    }

    pub fn reset(&mut self, batches: usize) {
        self.output_size[0] = batches
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        let output_size = IxDyn(&[inputs.shape()[0], self.output_size[1]]);
        inputs.into_shape_with_order(output_size).unwrap()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        let mut current_size = self.input_size.clone();
        current_size[0] = d_outputs.shape()[0];
        d_outputs.to_shape(current_size).unwrap().to_owned()
    }
}
