use ndarray::{Array1, Array2, ArrayD, Axis, Dimension, Ix1, Ix2, IxDyn};
use std::ops::{Add, Mul, SubAssign};

use crate::{CPUInit, DenseLayer, Init, Tensors};

pub struct DenseCPULayer {
    pub inputs: Array2<f32>,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub outputs: Array2<f32>,
}

impl DenseCPULayer {
    pub fn new(config: DenseLayer, size: IxDyn, tensors: Option<Tensors>) -> Self {
        let init = CPUInit::from_default(config.init, Init::Uniform);
        let input_size = Ix2(size[0], size[1]);
        let weight_size = Ix2(size[1], config.size[0]);
        let output_size = Ix2(size[0], config.size[0]);

        let (weights, biases) = if let Some(Tensors::Dense(tensors)) = tensors {
            (tensors.weights, tensors.biases)
        } else {
            let weights = init.init(weight_size.into_dyn(), size[1], output_size[1]);
            let biases = ArrayD::zeros(config.size);
            (weights, biases)
        };

        Self {
            inputs: Array2::zeros(input_size),
            weights: weights.into_dimensionality::<Ix2>().unwrap(),
            biases: biases.into_dimensionality::<Ix1>().unwrap(),
            outputs: Array2::zeros(output_size),
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.outputs.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let input_size = self.inputs.dim().1;
        self.inputs = Array2::zeros((batches, input_size));
        let output_size = self.outputs.dim().1;
        self.outputs = Array2::zeros((batches, output_size));
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix2>().unwrap();
        self.outputs = self.inputs.dot(&self.weights).add(&self.biases);
        self.outputs.clone().into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        let d_outputs = d_outputs.into_dimensionality::<Ix2>().unwrap();
        let mut weights_t = self.weights.view();
        weights_t.swap_axes(0, 1);
        let d_inputs = d_outputs.dot(&weights_t);
        let mut inputs_t = self.inputs.view();
        inputs_t.swap_axes(0, 1);
        let d_weights = inputs_t.dot(&d_outputs);
        self.weights.sub_assign(&d_weights.mul(rate));
        self.biases
            .sub_assign(&d_outputs.mul(rate).sum_axis(Axis(0)));
        d_inputs.into_dyn()
    }
}
