use std::ops::{Add, Div, Mul, SubAssign};

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::AdamOptimizer;

pub struct CPUAdamOptimizer {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m: Vec<Vec<ArrayD<f32>>>,
    pub v: Vec<Vec<ArrayD<f32>>>,
    pub t: f32,
}

impl CPUAdamOptimizer {
    pub fn new(config: AdamOptimizer, params: Vec<Vec<ArrayViewMutD<f32>>>) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for params in params {
            m.push(
                params
                    .iter()
                    .map(|param| ArrayD::zeros(param.dim()))
                    .collect(),
            );
            v.push(
                params
                    .iter()
                    .map(|param| ArrayD::zeros(param.dim()))
                    .collect(),
            );
        }
        Self {
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            m,
            v,
            t: 0.0,
        }
    }

    pub fn update_grads(
        &mut self,
        mut params: Vec<ArrayViewMutD<f32>>,
        grads: Vec<ArrayViewD<f32>>,
        idx: usize,
        rate: f32,
    ) {
        for (j, (param, grad)) in params.iter_mut().zip(grads).enumerate() {
            self.m[idx][j] = self
                .beta1
                .mul(&self.m[idx][j])
                .add((1.0 - self.beta1).mul(&grad));
            self.v[idx][j] = self
                .beta2
                .mul(&self.v[idx][j])
                .add((1.0 - self.beta2).mul(&grad.map(|x| x.powi(2))));

            let m_hat = self.m[idx][j].view().div(1.0 - self.beta1.powf(self.t));
            let v_hat = self.v[idx][j].view().div(1.0 - self.beta2.powf(self.t));

            param.sub_assign(
                &rate
                    .mul(m_hat)
                    .div(v_hat.map(|x| x.sqrt()).add(self.epsilon)),
            )
        }
    }
}
