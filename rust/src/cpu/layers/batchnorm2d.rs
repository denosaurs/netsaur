use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use ndarray::{Array4, ArrayD, Axis, Ix4, IxDyn};

use crate::{BatchNormLayer, Tensors};

macro_rules! axes {
    (($array:expr).sum_axes($($axis:literal),+)) => {
        $array$( .sum_axis(Axis($axis)).insert_axis(Axis($axis)) )+
    };
    (($array:expr).mean_axes($($axis:literal),+)) => {
        $array$( .mean_axis(Axis($axis)).unwrap().insert_axis(Axis($axis)) )+
    };
}

pub struct BatchNorm2DCPULayer {
    pub epsilon: f32,
    pub momentum: f32,

    // variables
    pub gamma: Array4<f32>,
    pub beta: Array4<f32>,
    pub running_mean: Array4<f32>,
    pub running_var: Array4<f32>,

    // cache
    pub inputs: Array4<f32>,
    pub mean: Array4<f32>,
    pub var: Array4<f32>,
    pub std_dev: Array4<f32>,
    pub normalized: Array4<f32>,
}

impl BatchNorm2DCPULayer {
    pub fn new(config: BatchNormLayer, size: IxDyn, tensors: Option<Tensors>) -> Self {
        let input_size = [size[0], size[1], size[2], size[3]];

        let (gamma, beta, running_mean, running_var) =
            if let Some(Tensors::BatchNorm(tensors)) = tensors {
                (
                    tensors.gamma.into_dimensionality().unwrap(),
                    tensors.beta.into_dimensionality().unwrap(),
                    tensors.running_mean.into_dimensionality().unwrap(),
                    tensors.running_var.into_dimensionality().unwrap(),
                )
            } else {
                (
                    Array4::ones((1, size[1], 1, 1)),
                    Array4::zeros((1, size[1], 1, 1)),
                    Array4::zeros((1, size[1], 1, 1)),
                    Array4::ones((1, size[1], 1, 1)),
                )
            };

        Self {
            epsilon: config.epsilon,
            momentum: config.momentum,

            gamma,
            beta,
            running_mean,
            running_var,
            
            inputs: Array4::zeros(input_size),
            mean: Array4::zeros((1, size[1], 1, 1)),
            var: Array4::zeros((1, size[1], 1, 1)),
            std_dev: Array4::zeros((1, size[1], 1, 1)),
            normalized: Array4::zeros(input_size),
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.inputs.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let size = self.inputs.shape();
        self.inputs = Array4::zeros((batches, size[1], size[2], size[3]));
        let size = self.inputs.shape();
        self.normalized = Array4::zeros((batches, size[1], size[2], size[3]));
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>, training: bool) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix4>().unwrap();

        if training {
            self.mean = axes!((self.inputs).mean_axes(0, 2, 3));
            self.var = axes!((self.inputs.map(|x| x.powi(2))).sum_axes(0, 2, 3))
                .div((self.inputs.len() / self.mean.len()) as f32)
                .sub(&self.mean.map(|x| x.powi(2)));
            self.running_mean = self
                .running_mean
                .view()
                .mul(self.momentum)
                .add(self.mean.view().mul(1.0 - self.momentum));
            self.running_var = self
                .running_var
                .view()
                .mul(self.momentum)
                .add(self.var.view().mul(1.0 - self.momentum));
        } else {
            self.mean = self.running_mean.clone();
            self.var = self.running_var.clone();
        };
        self.var.add_assign(self.epsilon);
        self.std_dev = self.var.map(|x| x.sqrt());
        self.normalized = self
            .inputs
            .view()
            .sub(&self.mean.view())
            .div(&self.std_dev.view());
        let batch_norm = self
            .gamma
            .view()
            .mul(&self.normalized.view())
            .add(self.beta.view());
        batch_norm.into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, rate: f32) -> ArrayD<f32> {
        let d_outputs = d_outputs.into_dimensionality::<Ix4>().unwrap();
        let output_y = self.inputs.shape()[2] as f32;
        let output_x = self.inputs.shape()[3] as f32;

        let mean_diff = self.inputs.view().sub(&self.mean.view());
        let d_normalized = d_outputs.view().mul(&self.gamma.view());
        let d_var = axes!((d_normalized
            .view()
            .mul(&mean_diff.view())
            .mul(-0.5)
            .mul(self.var.view().map(|x| x.powf(-1.5))))
        .sum_axes(0, 2, 3));
        let d_mean = &d_normalized
            .view()
            .mul(axes!(((-1.0).div(&self.std_dev.view())).sum_axes(0, 2, 3)))
            .add(
                d_var
                    .view()
                    .mul(&axes!(((-2.0).mul(&mean_diff.view())).sum_axes(0, 2, 3)))
                    .div(output_y * output_x),
            );

        self.gamma.sub_assign(
            &axes!((d_outputs.view().mul(&self.normalized.view())).sum_axes(0, 2, 3)).mul(rate),
        );
        self.beta
            .sub_assign(&axes!((d_outputs).sum_axes(0, 2, 3)).mul(rate));

        d_normalized
            .view()
            .div(&self.std_dev.view())
            .add(
                &d_var
                    .mul(2.0)
                    .mul(&mean_diff.view())
                    .div(output_y * output_x),
            )
            .add(d_mean.div(output_y * output_x))
            .into_dyn()
    }
}