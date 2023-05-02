use std::ops::{Sub, Mul};

use ndarray::{ArrayViewD, ArrayD};

use crate::Cost;

pub struct CPUCost {
    pub cost: for<'a> fn(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32,
    pub prime: for<'a> fn(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32>,
}

impl CPUCost {
    pub fn from(cost: Cost) -> CPUCost {
        match cost {
            Cost::MSE => CPUCost {
                cost: mse,
                prime: mse_prime,
            },
        }
    }
}

fn mse<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    let sub = y.sub(&y_hat);
    return sub.clone().mul(sub).sum();
}

fn mse_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    return y.sub(&y_hat);
}