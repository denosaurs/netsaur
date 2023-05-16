use std::ops::{Add, Div, Mul, Sub};

use ndarray::{Array4, ArrayD, Axis, Ix4, IxDyn};

use crate::BatchNormLayer;

pub struct BatchNorm2DCPULayer {
    pub epsilon: f32,
    pub initialized: bool,
    pub inputs: Array4<f32>,
    pub gamma: Array4<f32>,
    pub beta: Array4<f32>,
    pub running_mean: Array4<f32>,
    pub running_var: Array4<f32>,
    pub momentum: f32,
    pub iterations: f32,
}

impl BatchNorm2DCPULayer {
    pub fn new(config: BatchNormLayer, size: IxDyn) -> Self {
        let input_size = [size[0], size[1], size[2], size[3]];

        Self {
            epsilon: config.epsilon,
            initialized: false,
            inputs: Array4::zeros(input_size),
            gamma: Array4::ones((1, size[1], 1, 1)),
            beta: Array4::zeros((1, size[1], 1, 1)),
            running_mean: Array4::ones((1, size[1], 1, 1)),
            running_var: Array4::zeros((1, size[1], 1, 1)),
            momentum: config.momentum,
            iterations: 0.0,
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.inputs.shape().to_vec()
    }

    pub fn reset(&mut self, _batches: usize) {}

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>, training: bool) -> ArrayD<f32> {
        let inputs = inputs.into_dimensionality::<Ix4>().unwrap();

        self.iterations += 1.0;

        let (mean, var) = if training {
            let mean = inputs
                .mean_axis(Axis(0))
                .unwrap()
                .mean_axis(Axis(2))
                .unwrap()
                .mean_axis(Axis(3))
                .unwrap()
                .insert_axis(Axis(0))
                .insert_axis(Axis(2))
                .insert_axis(Axis(3));
            let var = inputs
                .var_axis(Axis(0), 0.0)
                .var_axis(Axis(2), 0.0)
                .var_axis(Axis(3), 0.0)
                .insert_axis(Axis(0))
                .insert_axis(Axis(2))
                .insert_axis(Axis(3));
            if self.initialized {
                self.running_mean = self
                    .running_mean
                    .view()
                    .mul(self.momentum / self.iterations)
                    .add(mean.view().mul(1.0 - self.momentum / self.iterations));
                self.running_var = self
                    .running_var
                    .view()
                    .mul(self.momentum / self.iterations)
                    .add(var.view().mul(1.0 - self.momentum / self.iterations));
            } else {
                self.running_mean = mean.clone();
                self.running_var = var.clone();
            }
            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };
        let normalized = inputs
            .sub(mean)
            .div(var.add(self.epsilon).map(|x| x.sqrt()));
        let batch_norm = self.gamma.view().mul(normalized).add(self.beta.view());
        batch_norm.into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        d_outputs
    }
}
