use ndarray::{Array1, Array2, ArrayD, Ix2, IxDyn, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::ops::{Add, Mul, AddAssign};

use crate::{CPUActivation, Dense};

pub struct DenseCPULayer {
    pub inputs: Array2<f32>,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub outputs: Array2<f32>,
    pub activation: Option<CPUActivation>,
}

impl DenseCPULayer {
    pub fn new(dense: Dense, size: IxDyn) -> Self {
        Self {
            inputs: Array2::zeros((size[0], size[1])),
            weights: Array2::random((size[1], dense.size[0]), Uniform::new(-1.0, 1.0)),
            biases: Array1::zeros(dense.size[0]),
            outputs: Array2::zeros((size[0], dense.size[0])),
            activation: CPUActivation::from_option(dense.activation),
        }
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
        if let Some(activation) = &self.activation {
            self.outputs = self.outputs.map(activation.activate)
        };
        self.outputs.clone().into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        let mut d_outputs = d_outputs.into_dimensionality::<Ix2>().unwrap();
        if let Some(activation) = &self.activation {
            d_outputs = d_outputs.mul(self.outputs.map(activation.prime))
        }
        let mut weights_t = self.weights.view();
        weights_t.swap_axes(0, 1);
        let d_inputs = d_outputs.dot(&weights_t);
        let mut inputs_t = self.inputs.view();
        inputs_t.swap_axes(0, 1);
        let d_weights = inputs_t.dot(&d_outputs);
        self.weights.add_assign(&d_weights.mul(rate));
        self.biases.add_assign(&d_outputs.mul(rate).sum_axis(Axis(0)));
        d_inputs.into_dyn()
    }
}
