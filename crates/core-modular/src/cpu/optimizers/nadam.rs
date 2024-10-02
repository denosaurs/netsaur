use std::ops::{Add, Div, Mul, SubAssign, Sub};

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::NadamOptimizer;

pub struct CPUNadamOptimizer {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m: Vec<Vec<ArrayD<f32>>>,
    pub n: Vec<Vec<ArrayD<f32>>>,
    pub t: f32,
}

impl CPUNadamOptimizer {
    pub fn new(config: NadamOptimizer, params: Vec<Vec<ArrayViewMutD<f32>>>) -> Self {
        let mut m = Vec::new();
        let mut n = Vec::new();
        for params in params {
            m.push(
                params
                    .iter()
                    .map(|param| ArrayD::zeros(param.dim()))
                    .collect(),
            );
            n.push(
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
            n,
            t: 0.0,
        }
    }

    pub fn update_grads(
        &mut self,
        mut params: Vec<ArrayViewMutD<f32>>,
        grads: Vec<ArrayViewD<f32>>,
        idx: usize,
        rate: f32,
        l: Vec<ArrayViewD<f32>>,
    ) {
        for (j, ((param, grad), li)) in params.iter_mut().zip(grads).zip(l).enumerate() {
            self.m[idx][j] = self
                .beta1
                .mul(&self.m[idx][j])
                .add((1.0 - self.beta1).mul(&grad));
            self.n[idx][j] = self
                .beta2
                .mul(&self.n[idx][j])
                .add((1.0 - self.beta2).mul(&grad.map(|x| x.powi(2))));

            let m_hat = self.m[idx][j].view();
            let n_hat = self.n[idx][j].view().div(1.0 - self.beta2.powf(self.t));

            let nestrov_m_hat = self.beta1.mul(&m_hat).add((1.0 - self.beta1).mul(&grad)).div(1.0 - self.beta1.powf(self.t));

            param.sub_assign(
                &rate
                    .mul(nestrov_m_hat)
                    .div(n_hat.map(|x| x.sqrt()).add(self.epsilon))
                    .sub(&li),
            )
        }
    }
}
