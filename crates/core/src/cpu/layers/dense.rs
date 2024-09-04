use ndarray::{Array1, Array2, ArrayD, Axis, Dimension, Ix1, Ix2, IxDyn};
use std::ops::Add;

use crate::{CPUInit, CPURegularizer, DenseLayer, Init, Tensors};

pub struct DenseCPULayer {
    // cache
    pub output_size: Ix2,
    pub inputs: Array2<f32>,

    // parameters
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,

    // gradients
    pub d_weights: Array2<f32>,
    pub d_biases: Array1<f32>,

    // regularization
    pub l_weights: Array2<f32>,
    pub l_biases: Array1<f32>,

    pub regularizer: CPURegularizer,
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
            let weights = init.init(weight_size.into_dyn(), size[1], config.size[0]);
            let biases = ArrayD::zeros(config.size.clone());
            (weights, biases)
        };

        Self {
            output_size,
            inputs: Array2::zeros(input_size),
            weights: weights.into_dimensionality::<Ix2>().unwrap(),
            biases: biases.into_dimensionality::<Ix1>().unwrap(),
            d_weights: Array2::zeros(weight_size),
            d_biases: Array1::zeros(config.size[0]),
            l_weights: Array2::zeros(weight_size),
            l_biases: Array1::zeros(config.size[0]),
            regularizer: CPURegularizer::from(config.c, config.l1_ratio)
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.output_size.as_array_view().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let input_size = self.inputs.dim().1;
        self.inputs = Array2::zeros((batches, input_size));
        self.output_size[0] = batches;
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix2>().unwrap();
        self.inputs.dot(&self.weights).add(&self.biases).into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        let d_outputs = d_outputs.into_dimensionality::<Ix2>().unwrap();
        let mut weights_t = self.weights.view();
        weights_t.swap_axes(0, 1);
        let d_inputs = d_outputs.dot(&weights_t);
        let mut inputs_t = self.inputs.view();
        inputs_t.swap_axes(0, 1);
        self.d_weights = inputs_t.dot(&d_outputs);
        self.d_biases = d_outputs.sum_axis(Axis(0));

        self.l_weights = self.regularizer.coeff(&self.weights.clone().into_dyn()).into_dimensionality::<Ix2>().unwrap();
        self.l_biases = self.regularizer.coeff(&self.biases.clone().into_dyn()).into_dimensionality::<Ix1>().unwrap();
        d_inputs.into_dyn()
    }
}
