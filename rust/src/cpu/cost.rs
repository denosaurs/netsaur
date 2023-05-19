use std::ops::{Div, Mul, Sub};

use ndarray::{ArrayD, ArrayViewD};

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
            Cost::CrossEntropy => CPUCost {
                cost: cross_entropy,
                prime: cross_entropy_prime,
            },
            Cost::Hinge => CPUCost {
                cost: hinge,
                prime: hinge_prime,
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

fn cross_entropy<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    return -y_hat.mul(&y).sum().ln();
}

fn cross_entropy_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    return -y_hat.div(&y);
}

fn hinge<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    let mut sum = 0.0;
    for (y_hat_i, y_i) in y_hat.iter().zip(y.iter()) {
        let margin = 1.0 - y_hat_i * y_i;
        if margin > 0.0 {
            sum += margin;
        }
    }
    return sum;
}

fn hinge_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    let mut result = ArrayD::zeros(y_hat.shape());
    for ((result_i, y_hat_i), y_i) in result.iter_mut().zip(y_hat.iter()).zip(y.iter()) {
        let margin = 1.0 - y_hat_i * y_i;
        if margin > 0.0 {
            *result_i = -y_i;
        }
    }
    return result;
}
