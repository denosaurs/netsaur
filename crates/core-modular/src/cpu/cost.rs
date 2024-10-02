use std::{
    f32::EPSILON,
    ops::{Mul, Sub},
};

use ndarray::{Array1, ArrayD, ArrayViewD};

use crate::Cost;

const HUBER_DELTA: f32 = 1.5;
const TUKEY_C: f32 = 4.685;

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
            Cost::MAE => CPUCost {
                cost: mae,
                prime: mae_prime,
            },
            Cost::CrossEntropy => CPUCost {
                cost: cross_entropy,
                prime: cross_entropy_prime,
            },
            Cost::BinCrossEntropy => CPUCost {
                cost: bin_cross_entropy,
                prime: bin_cross_entropy_prime,
            },
            Cost::Hinge => CPUCost {
                cost: hinge,
                prime: hinge_prime,
            },
            Cost::Huber => CPUCost {
                cost: huber,
                prime: huber_prime,
            },
            Cost::SmoothHinge => CPUCost {
                cost: smooth_hinge,
                prime: smooth_hinge_prime,
            },
            Cost::Tukey => CPUCost {
                cost: tukey,
                prime: tukey_prime,
            },
        }
    }
}

fn mse<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    let sub = y_hat.sub(&y);
    return sub.clone().mul(sub).sum() / y.len() as f32;
}

fn mse_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    return y_hat.sub(&y);
}

fn mae<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    let sub = y_hat.sub(&y);
    return sub.map(|x| x.abs()).sum() / y.len() as f32;
}

fn mae_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    return y_hat.sub(&y).map(|x| x.signum());
}

fn cross_entropy<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    let batches = y_hat.dim()[0];
    let total = (-&y * (y_hat.map(|x| x.max(EPSILON).min(1f32 - EPSILON).ln()))).sum();
    return total / batches as f32;
}

fn cross_entropy_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    return -&y / (&y_hat + EPSILON);
}

fn bin_cross_entropy<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    return -y_hat
        .mul(y.map(|x| x.max(EPSILON).min(1f32 - EPSILON).ln()))
        .sub(((1.0).sub(&y_hat)).mul(y.map(|x| 1.0 - x.max(EPSILON).min(1f32 - EPSILON).ln())))
        .sum()
        / y.len() as f32;
}

fn bin_cross_entropy_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    return (-&y / (&y_hat + EPSILON)) + (1.0 - &y) / (1.0 - &y_hat + EPSILON);
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
    return result
}

pub fn smooth_hinge<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    y_hat
        .iter()
        .zip(y.iter())
        .map(|(y_hat_i, y_i)| {
            let margin = y_i * y_hat_i;
            if margin > -1f32 {
                (1.0 - margin).max(0.0)
            } else {
                -4f32 * margin
            }
        })
        .collect::<Array1<f32>>()
        .to_shape(y.shape())
        .unwrap()
        .to_owned()
        .sum()
        / y.len() as f32
}

pub fn smooth_hinge_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    y_hat
        .iter()
        .zip(y.iter())
        .map(|(y_hat_i, y_i)| {
            let margin = y_i * y_hat_i;
            if margin > -1f32 {
                -y_i
            } else {
                -4f32 * y_i
            }
        })
        .collect::<Array1<f32>>()
        .to_shape(y.shape())
        .unwrap()
        .to_owned()
}

pub fn tukey<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    let c_squared = TUKEY_C * TUKEY_C / 6.0;
    y.sub(&y_hat)
        .map(|el| {
            let r = el.abs();
            if r <= TUKEY_C {
                c_squared * (1.0 - (1.0 - (r / TUKEY_C).powi(2)).powi(3))
            } else {
                c_squared
            }
        })
        .sum()
        / y.len() as f32
}

pub fn tukey_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    y.sub(&y_hat).map(|el| {
        let r = el.abs();
        if r <= TUKEY_C {
            r * (1.0 - ((r / TUKEY_C).powi(2))).powi(2)
        } else {
            0f32
        }
    })
}

pub fn huber<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> f32 {
    let loss: Array1<f32> = y_hat
        .iter()
        .zip(y.iter())
        .map(|(y_hat_i, y_i)| {
            let residual = y_i - y_hat_i;
            if residual.abs() <= HUBER_DELTA {
                0.5 * residual.powi(2)
            } else {
                HUBER_DELTA * (residual.abs() - 0.5 * HUBER_DELTA)
            }
        })
        .collect();
    loss.to_shape(y.shape()).unwrap().sum() / y.len() as f32
}

pub fn huber_prime<'a>(y_hat: ArrayViewD<'a, f32>, y: ArrayViewD<'a, f32>) -> ArrayD<f32> {
    let gradient: Array1<f32> = y_hat
        .iter()
        .zip(y.iter())
        .map(|(y_hat_i, y_i)| {
            let residual = y_i - y_hat_i;
            if residual.abs() <= HUBER_DELTA {
                -residual
            } else {
                -HUBER_DELTA * residual.signum()
            }
        })
        .collect();
    gradient.to_shape(y.shape()).unwrap().to_owned()
}
