use ndarray::{Array1, Array2, ArrayD, Axis, Dimension, Ix2, IxDyn};
use std::ops::{Add, AddAssign, Mul};

use crate::{CPUActivation, CPUInit, DenseLayer, Init};

pub struct DenseCPULayer {
    pub inputs: Array2<f32>,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub outputs: Array2<f32>,
    pub activation: Option<CPUActivation>,
}

impl DenseCPULayer {
    pub fn new(config: DenseLayer, size: IxDyn) -> Self {
        let init = CPUInit::from_default(config.init, Init::Uniform);
        let input_size = Ix2(size[0], size[1]);
        let weights_size = Ix2(size[1], config.size[0]);
        let output_size = Ix2(size[0], config.size[0]);
        Self {
            inputs: Array2::zeros(input_size),
            weights: init
                .init(weights_size.into_dyn(), size.size(), output_size.size())
                .into_dimensionality::<Ix2>()
                .unwrap(),
            biases: Array1::zeros(config.size[0]),
            outputs: Array2::zeros(output_size),
            activation: CPUActivation::from_option(config.activation),
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
        let mut outputs = self.inputs.dot(&self.weights).add(&self.biases);
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
        self.biases
            .add_assign(&d_outputs.mul(rate).sum_axis(Axis(0)));
        d_inputs.into_dyn()
    }
}
