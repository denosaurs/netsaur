use ndarray::{ArrayD, Dimension, IxDyn};

use crate::FlattenLayer;

pub struct FlattenCPULayer {
    pub input_size: IxDyn,
    pub output_size: Vec<usize>,
}

impl FlattenCPULayer {
    pub fn new(config: FlattenLayer, size: IxDyn) -> Self {
        let mut new_size = config.size.clone();
        new_size.insert(0, size[0]);
        let output_size = IxDyn(&new_size);
        if output_size.size() != size.size() {
            panic!(
                "Shape {:#?} is incompatible with shape {:#?}",
                output_size, size
            )
        }
        Self {
            input_size: size,
            output_size: new_size,
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.output_size.clone()
    }

    pub fn reset(&mut self, batches: usize) {
        self.output_size[0] = batches
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        let output_size = IxDyn(&self.output_size);
        inputs.into_shape_with_order(output_size).unwrap()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        d_outputs.into_shape_with_order(self.input_size.clone()).unwrap()
    }
}
