use std::ops::Mul;

use ndarray::{Array2, Array4, ArrayD, Axis, IxDyn};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::DropoutLayer;

pub struct Dropout1DCPULayer {
    mask: ArrayD<f32>,
    probability: f32,
}

impl Dropout1DCPULayer {
    pub fn new(config: DropoutLayer, size: IxDyn) -> Self {
        Self {
            mask: ArrayD::zeros(size),
            probability: config.probability,
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.mask.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let mut output_size = self.mask.shape().to_vec();
        output_size[0] = batches;
        self.mask = ArrayD::zeros(output_size);
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.mask = ArrayD::random(self.mask.dim(), Uniform::new(0.0, 1.0))
            .map(|x| (if x > &self.probability { 1.0 } else { 0.0 }));
        inputs.mul(&self.mask).mul(1.0 / 1.0 - self.probability)
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        d_outputs.mul(&self.mask).mul(1.0 / 1.0 - self.probability)
    }
}

pub struct Dropout2DCPULayer {
    mask: Array4<f32>,
    probability: f32,
}

impl Dropout2DCPULayer {
    pub fn new(config: DropoutLayer, size: IxDyn) -> Self {
        Self {
            mask: Array4::zeros([size[0], size[1], size[2], size[3]]),
            probability: config.probability,
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.mask.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let size = self.mask.dim();
        self.mask = Array4::zeros([batches, size.1, size.2, size.3]);
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>, training: bool) -> ArrayD<f32> {
        if training {
            let size = self.mask.dim();
            self.mask = Array2::random([size.0, size.1], Uniform::new(0.0, 1.0))
                .map(|x| (if x > &self.probability { 1.0 } else { 0.0 }))
                .insert_axis(Axis(2))
                .insert_axis(Axis(3))
                .broadcast(size)
                .unwrap()
                .to_owned();
            inputs.mul(&self.mask).mul(1.0 / 1.0 - self.probability)
        } else {
            inputs
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        d_outputs.mul(&self.mask).mul(1.0 / 1.0 - self.probability)
    }
}
