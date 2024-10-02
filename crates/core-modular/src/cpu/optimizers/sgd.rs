use std::ops::{Mul, SubAssign};

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
        epoch: usize,
        l: Vec<ArrayViewD<f32>>,
    ) {
        for ((param, grad), li) in params.iter_mut().zip(grads).zip(l) {
            param.sub_assign(&(&grad - &li).mul(rate));
        }
    }
}
