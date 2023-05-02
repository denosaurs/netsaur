use crate::Activation;
pub struct CPUActivation {
    pub activate: fn(x: &f32) -> f32,
    pub prime: fn(x: &f32) -> f32,
}

impl CPUActivation {
    pub fn from(activation: Activation) -> CPUActivation {
        match activation {
            Activation::Sigmoid => CPUActivation {
                activate: sigmoid,
                prime: sigmoid_prime,
            },
            Activation::Tanh => CPUActivation {
                activate: tanh,
                prime: tanh_prime,
            },
        }
    }

    pub fn from_option(activation: Option<Activation>) -> Option<CPUActivation> {
        if let Some(activation) = activation {
            Some(CPUActivation::from(activation))
        } else {
            None
        }
    }
}

fn sigmoid(x: &f32) -> f32 {
    return 1.0 / (1.0 + (-x).exp());
}

fn sigmoid_prime(x: &f32) -> f32 {
    return x * (1.0 - x);
}

fn tanh(x: &f32) -> f32 {
    return x.tanh();
}

fn tanh_prime(x: &f32) -> f32 {
    return 1.0 - tanh(x).powi(2);
}
