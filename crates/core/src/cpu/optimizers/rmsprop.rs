use std::ops::{Add, Div, Mul, SubAssign, Sub};

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::{CPUScheduler, RMSPropOptimizer};

pub struct CPURMSPropOptimizer {
    decay_rate: f32,
    epsilon: f32,
    acc_sg: Vec<Vec<ArrayD<f32>>>,
}

impl CPURMSPropOptimizer {
    pub fn new(config: RMSPropOptimizer, params: Vec<Vec<ArrayViewMutD<f32>>>) -> Self {
        let mut acc_sg = Vec::new();
        for params in params {
            acc_sg.push(
                params
                    .iter()
                    .map(|param| ArrayD::zeros(param.dim()))
                    .collect(),
            );
        }
        Self {
            acc_sg,
            decay_rate: config.decay_rate,
            epsilon: config.epsilon,
        }
    }

    pub fn update_grads(
        &mut self,
        mut params: Vec<ArrayViewMutD<f32>>,
        grads: Vec<ArrayViewD<f32>>,
        idx: usize,
        scheduler: &CPUScheduler,
        rate: f32,
        epoch: usize,
        l: Vec<ArrayViewD<f32>>,
    ) {
        for (j, ((param, grad), li)) in params.iter_mut().zip(grads).zip(l).enumerate() {
            self.acc_sg[idx][j] = self
                .decay_rate
                .mul(&self.acc_sg[idx][j])
                .add((1.0 - self.decay_rate).mul(&grad.map(|x| x.powi(2))));

            let rate = scheduler.eta(rate, epoch);

            param.sub_assign(
                &rate
                    .mul(&grad)
                    .div(self.acc_sg[idx][j].map(|x| x.sqrt()).add(self.epsilon))
                    .sub(&li),
            )
        }
    }
}
