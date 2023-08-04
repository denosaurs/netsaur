use std::ops::{SubAssign, Mul};

use ndarray::{ArrayViewD, ArrayViewMutD};

pub struct CPUSGDOptimizer {}

impl CPUSGDOptimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn update_grads(
        &mut self,
        mut params: Vec<ArrayViewMutD<f32>>,
        grads: Vec<ArrayViewD<f32>>,
        rate: f32,
    ) {
        for (param, grad) in params.iter_mut().zip(grads) {
            param.sub_assign(&grad.mul(rate))
        }
    }
}
