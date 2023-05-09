use ndarray::{s, Array1, Array4, ArrayD, Dimension, IntoDimension, Ix4, IxDyn};
use std::ops::{Add, Mul};

use crate::{CPUActivation, CPUInit, Conv2DLayer, Init};

pub struct Conv2DCPULayer {
    pub strides: Vec<usize>,
    pub padding: usize,
    pub inputs: Array4<f32>,
    pub weights: Array4<f32>,
    pub biases: Array1<f32>,
    pub outputs: Array4<f32>,
    pub activation: Option<CPUActivation>,
}

impl Conv2DCPULayer {
    pub fn new(config: Conv2DLayer, size: IxDyn) -> Self {
        let strides = config.strides.unwrap_or(vec![1, 1]);
        let output_y = 1 + (size[2] + 2 * config.padding - config.kernel_size[2]) / strides[0];
        let output_x = 1 + (size[3] + 2 * config.padding - config.kernel_size[3]) / strides[1];
        let input_size = Ix4(size[0], size[1], size[2], size[3]);
        let weights_size = config.kernel_size.into_dimension();
        let output_size = Ix4(size[0], weights_size[0], output_y, output_x);
        let weights = if let Some(tensor) = config.kernel {
            let shape: [usize; 4] = tensor.shape.try_into().unwrap();
            Array4::from_shape_vec(shape, tensor.data).unwrap()
        } else {
            CPUInit::from_default(config.init, Init::Kaiming)
                .init(weights_size, input_size.size(), output_size.size())
                .into_dimensionality::<Ix4>()
                .unwrap()
        };
        Self {
            strides,
            padding: config.padding,
            inputs: Array4::zeros(input_size),
            weights,
            biases: Array1::zeros(size[0]),
            outputs: Array4::zeros(output_size),
            activation: CPUActivation::from_option(config.activation),
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.outputs.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let input_size = self.inputs.shape();
        self.inputs = Array4::zeros((batches, input_size[1], input_size[2], input_size[3]));
        let output_size = self.outputs.shape();
        self.outputs = Array4::zeros((batches, output_size[1], output_size[2], output_size[3]));
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix4>().unwrap();

        let (filters, _, weights_y, weights_x) = self.weights.dim();
        let (batches, _, outputs_y, outputs_x) = self.outputs.dim();

        let unpadded_y = outputs_y - self.padding * 2;
        let unpadded_x = outputs_x - self.padding * 2;

        let mut outputs = Array4::zeros(self.outputs.dim());
        for b in 0..batches {
            for f in 0..filters {
                let mut h = self.padding;
                for y in (0..unpadded_y).step_by(self.strides[0]) {
                    let mut w = self.padding;
                    for x in (0..unpadded_x).step_by(self.strides[1]) {
                        outputs[(b, f, h, w)] = self
                            .inputs
                            .slice(s![b, .., y..(y + weights_y), x..(x + weights_x)])
                            .mul(&self.weights.slice(s![f, .., .., ..]))
                            .sum()
                            .add(self.biases[f]);
                        w += 1;
                    }
                    h += 1;
                }
            }
        }

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
