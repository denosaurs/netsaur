use std::ops::{SubAssign, Mul};

use ndarray::{ArrayViewD, ArrayViewMutD};

use crate::CPUScheduler;

pub struct CPUSGDOptimizer {}

impl CPUSGDOptimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn update_grads(
        &mut self,
        mut params: Vec<ArrayViewMutD<f32>>,
        grads: Vec<ArrayViewD<f32>>,
        scheduler: &CPUScheduler,
        rate: f32,
        epoch: usize,
    ) {
        let eta = scheduler.eta(rate, epoch);
        for (param, grad) in params.iter_mut().zip(grads) {
            param.sub_assign(&grad.mul(eta));
        }
    }
}
