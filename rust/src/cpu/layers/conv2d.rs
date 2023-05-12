use ndarray::{s, Array1, Array4, ArrayD, Dimension, Ix1, Ix4, IxDyn};
use std::ops::{Add, AddAssign, Mul};

use crate::{CPUInit, Conv2DLayer, Init};

pub struct Conv2DCPULayer {
    pub strides: Vec<usize>,
    pub padding: Vec<usize>,
    pub inputs: Array4<f32>,
    pub weights: Array4<f32>,
    pub biases: Array1<f32>,
    pub outputs: Array4<f32>,
}

impl Conv2DCPULayer {
    pub fn new(
        config: Conv2DLayer,
        size: IxDyn,
        weights: Option<ArrayD<f32>>,
        biases: Option<ArrayD<f32>>,
    ) -> Self {
        let strides = config.strides.unwrap_or(vec![1, 1]);
        let padding = config.padding.unwrap_or(vec![0, 0]);
        let input_y = size[2] + 2 * padding[0];
        let input_x = size[3] + 2 * padding[1];
        let output_y = 1 + (input_y - config.kernel_size[2]) / strides[0];
        let output_x = 1 + (input_x - config.kernel_size[3]) / strides[1];
        let input_size = Ix4(size[0], size[1], input_y, input_x);
        let weight_size = IxDyn(config.kernel_size.as_slice());
        let output_size = Ix4(size[0], weight_size[0], output_y, output_x);

        let weights = weights.unwrap_or(if let Some(tensor) = config.kernel {
            ArrayD::from_shape_vec(tensor.shape, tensor.data).unwrap()
        } else {
            CPUInit::from_default(config.init, Init::Kaiming).init(
                weight_size,
                input_size.size(),
                output_size.size(),
            )
        });
        let biases = biases.unwrap_or(ArrayD::zeros(vec![config.kernel_size[0]]));

        Self {
            strides,
            padding,
            inputs: Array4::zeros(input_size),
            weights: weights.into_dimensionality::<Ix4>().unwrap(),
            biases: biases.into_dimensionality::<Ix1>().unwrap(),
            outputs: Array4::zeros(output_size),
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
        let inputs = inputs.into_dimensionality::<Ix4>().unwrap();
        let (_, _, input_y, input_x) = self.inputs.dim();
        let unpadded_y = self.padding[0]..input_y - self.padding[0];
        let unpadded_x = self.padding[1]..input_x - self.padding[1];
        self.inputs
            .slice_mut(s![.., .., unpadded_y, unpadded_x])
            .assign(&inputs);

        let (filters, _, weight_y, weight_x) = self.weights.dim();
        let (batches, _, output_y, output_x) = self.outputs.dim();

        for b in 0..batches {
            for f in 0..filters {
                let mut h = 0;
                for y in (0..output_y).step_by(self.strides[0]) {
                    let mut w = 0;
                    for x in (0..output_x).step_by(self.strides[1]) {
                        self.outputs[(b, f, h, w)] = self
                            .inputs
                            .slice(s![b, .., y..y + weight_y, x..x + weight_x])
                            .mul(&self.weights.slice(s![f, .., .., ..]))
                            .sum()
                            .add(self.biases[f]);
                        w += 1;
                    }
                    h += 1;
                }
            }
        }

        self.outputs.clone().into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        let d_outputs = d_outputs.into_dimensionality::<Ix4>().unwrap();

        let (filters, _, weight_y, weight_x) = self.weights.dim();
        let (batches, _, output_y, output_x) = self.outputs.dim();

        let mut d_inputs = Array4::zeros(self.inputs.dim());
        let mut d_weights = Array4::zeros(self.weights.dim());
        let mut d_biases = Array1::<f32>::zeros(self.biases.dim());
        for b in 0..batches {
            for f in 0..filters {
                for y in (0..output_y).step_by(self.strides[0]) {
                    for x in (0..output_x).step_by(self.strides[1]) {
                        d_inputs
                            .slice_mut(s![b, .., y..y + weight_y, x..x + weight_x])
                            .add_assign(
                                &self
                                    .weights
                                    .slice(s![f, .., .., ..])
                                    .mul(d_outputs[(b, f, y, x)]),
                            );
                        d_weights.slice_mut(s![f, .., .., ..]).add_assign(
                            &self
                                .inputs
                                .slice(s![b, .., y..y + weight_y, x..x + weight_x])
                                .mul(d_outputs[(b, f, y, x)]),
                        );
                        d_biases[f] += d_outputs[(b, f, y, x)];
                    }
                }
            }
        }

        self.weights.add_assign(&d_weights.mul(rate));
        self.biases.add_assign(&d_biases.mul(rate));

        d_inputs.into_dyn()
    }
}
