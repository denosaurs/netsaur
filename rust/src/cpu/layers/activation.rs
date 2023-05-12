use ndarray::{ArrayD, IxDyn};
use std::ops::{Mul, Div};

use crate::{CPUActivation, ActivationLayer};

pub struct ActivationCPULayer {
    pub outputs: ArrayD<f32>,
    pub activation: CPUActivation,
}

impl ActivationCPULayer {
    pub fn new(config: ActivationLayer, size: IxDyn) -> Self {
        Self {
            outputs: ArrayD::zeros(size),
            activation: CPUActivation::from(config.activation),
        }
    }
    
    pub fn output_size(&self) -> Vec<usize> {
        self.outputs.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let mut output_size = self.outputs.shape().to_vec();
        output_size[0] = batches;
        self.outputs = ArrayD::zeros(output_size);
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        let outputs = if CPUActivation::memoize_output(&self.activation) {
            self.outputs = inputs.map(self.activation.activate);
            self.outputs.clone()
        } else {
            self.outputs = inputs.clone();
            inputs.map(self.activation.activate)
        };
        outputs.into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        let d_inputs = d_outputs.mul(self.outputs.map(self.activation.prime));
        d_inputs.into_dyn()
    }
}

pub struct SoftmaxCPULayer {
    pub outputs: ArrayD<f32>,
}

impl SoftmaxCPULayer {
    pub fn new(size: IxDyn) -> Self {
        Self {
            outputs: ArrayD::zeros(size),
        }
    }
    
    pub fn output_size(&self) -> Vec<usize> {
        self.outputs.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let mut output_size = self.outputs.shape().to_vec();
        output_size[0] = batches;
        self.outputs = ArrayD::zeros(output_size);
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.outputs = inputs.map(|x| x.exp()).div(inputs.map(|x| x.exp()).sum());
        self.outputs.clone().into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        d_outputs
    }
}