use std::ops::{Mul, SubAssign};

use ndarray::{ArrayViewD, ArrayViewMutD};

use crate::GPUScheduler;

pub struct GPUSGDOptimizer {}

impl GPUSGDOptimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn update_grads(
        &mut self,
        mut params: Vec<ArrayViewMutD<f32>>,
        grads: Vec<ArrayViewD<f32>>,
        scheduler: &GPUScheduler,
        rate: f32,
        epoch: usize,
    ) {
        let eta = scheduler.eta(rate, epoch);
        for (param, grad) in params.iter_mut().zip(grads) {
            param.sub_assign(&grad.mul(eta));
        }
    }
}
