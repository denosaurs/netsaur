use ndarray::{Array4, Array2, ArrayD, IxDyn};
use std::ops::{Add, AddAssign, Mul};

use crate::{CPUActivation, CPUInit, Conv2DLayer, Init};

pub struct Conv2DCPULayer {
    pub inputs: Array4<f32>,
    pub weights: Array4<f32>,
    pub biases: Array2<f32>,
    pub outputs: Array4<f32>,
    pub activation: CPUActivation,
}

impl Conv2DCPULayer {
    pub fn new(config: Conv2DLayer, size: IxDyn) -> Self {
        let init = CPUInit::from_default(config.init, Init::Kaiming);
        let output_y = 1 + (size[2] + 2 * config.padding - config.kernel_size[2]) / config.stride;
        let output_x = 1 + (size[3] + 2 * config.padding - config.kernel_size[3]) / config.stride;
        let output_size = [size[0], size[1], output_y, output_x];
        Self {
            inputs: Array4::zeroes(size),
            weights: (init.init)(&weights_size, &input_size, &output_size)
                .into_dimensionality::<Ix4>()
                .unwrap(),
            biases: Array2::zeroes([size[0], size[1]]),
            outputs: Array4::zeroes(output_size),
            activation: CPUActivation::from_option(config.activation),
        }
    }

    pub fn reset(&mut self, batches: usize) {
        let input_size = self.inputs.shape();
        self.inputs = Array2::zeros((batches, input_size[1], input_size[2], input_size[3]));
        let output_size = self.outputs.shape();
        self.outputs = Array2::zeros((batches, output_size[1], output_size[2], output_size[3]));
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix2>().unwrap();
        // todo feedforward
        let mut outputs = self.inputs;
        if let Some(activation) = &self.activation {
            if CPUActivation::memoize_output(activation) {
                outputs = outputs.map(activation.activate);
                self.outputs = outputs.clone();
            } else {
                self.outputs = outputs.clone();
                outputs = outputs.map(activation.activate);
            }
        };
        outputs.into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        d_outputs
    }
}
