use ndarray::{Array2, Array4, ArrayD, Ix2, Ix4, IxDyn};
use std::ops::{Add, AddAssign, Mul};

use crate::{CPUActivation, CPUInit, Conv2DLayer, Init};

pub struct Conv2DCPULayer {
    pub inputs: Array4<f32>,
    pub weights: Array4<f32>,
    pub biases: Array2<f32>,
    pub outputs: Array4<f32>,
    pub activation: Option<CPUActivation>,
}

impl Conv2DCPULayer {
    pub fn new(config: Conv2DLayer, size: IxDyn) -> Self {
        let init = CPUInit::from_default(config.init, Init::Kaiming);
        let mut strides = config.strides.unwrap_or(vec![1, 1]);
        if strides.len() != 2 {
            strides = vec![1, 1];
        }
        let output_y = 1 + (size[2] + 2 * config.padding - config.kernel_size[2]) / strides[0];
        let output_x = 1 + (size[3] + 2 * config.padding - config.kernel_size[3]) / strides[1];
        let output_size = [size[0], size[1], output_y, output_x];
        Self {
            inputs: Array4::zeros(size),
            weights: (init.init)(&weights_size, &input_size, &output_size)
                .into_dimensionality::<Ix4>()
                .unwrap(),
            biases: Array2::zeros([size[0], size[1]]),
            outputs: Array4::zeros(output_size),
            activation: CPUActivation::from_option(config.activation),
        }
    }

    pub fn reset(&mut self, _batches: usize) {}

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix4>().unwrap();
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
