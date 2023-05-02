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
            Activation::Linear => CPUActivation {
                activate: linear,
                prime: linear_prime,
            },
            Activation::Relu => CPUActivation {
                activate: relu,
                prime: relu_prime,
            },
            Activation::Relu6 => CPUActivation {
                activate: relu6,
                prime: relu6_prime,
            },
            Activation::LeakyRelu => CPUActivation {
                activate: leaky_relu,
                prime: leaky_relu_prime,
            },
            Activation::Elu => CPUActivation {
                activate: elu,
                prime: elu_prime,
            },
            Activation::Selu => CPUActivation {
                activate: selu,
                prime: selu_prime,
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

fn linear(x: &f32) -> f32 {
    return *x;
}

fn linear_prime(_x: &f32) -> f32 {
    return 1.0;
}

fn relu(x: &f32) -> f32 {
    return x.max(0.0);
}

fn relu_prime(x: &f32) -> f32 {
    return if *x > 0.0 { 1.0 } else { 0.0 };
}

fn relu6(x: &f32) -> f32 {
    return x.max(0.0).min(6.0);
}

fn relu6_prime(x: &f32) -> f32 {
    return if *x > 0.0 { 1.0 } else { 0.0 };
}

fn leaky_relu(x: &f32) -> f32 {
    return if *x > 0.0 { *x } else { x.max(0.01 * x) };
}

fn leaky_relu_prime(x: &f32) -> f32 {
    return if *x > 0.0 { 1.0 } else { 0.01 };
}

fn elu(x: &f32) -> f32 {
    return if *x >= 0.0 { *x } else { x.exp() - 1.0 };
}

fn elu_prime(x: &f32) -> f32 {
    return if *x > 0.0 { 1.0 } else { x.exp() };
}

fn selu(x: &f32) -> f32 {
    return if *x >= 0.0 { *x } else { 1.0507 * (x.exp() - 1.0) };
}

fn selu_prime(x: &f32) -> f32 {
    return if *x > 0.0 { 1.0 } else { 1.0507 * x.exp() };
}