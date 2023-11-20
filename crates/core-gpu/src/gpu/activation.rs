use crate::Activation;
pub struct GPUActivation {
    pub activation: Activation,
    pub activate: String,
    pub prime: String,
}

impl GPUActivation {
    pub fn from(activation: Activation) -> Self {
        let (activate, prime): (&str, &str) = match activation {
            Activation::Sigmoid => (SIGMOID, SIGMOID_PRIME),
            _ => unimplemented!()
        };

        Self {
            activation,
            activate: String::from(activate),
            prime: String::from(prime),
        }
    }

    pub fn from_option(activation: Option<Activation>) -> Option<GPUActivation> {
        if let Some(activation) = activation {
            Some(GPUActivation::from(activation))
        } else {
            None
        }
    }

    pub fn memoize_output(activation: &Activation) -> bool {
        match activation {
            Activation::Sigmoid | Activation::Tanh => true,
            _ => true,
        }
    }
}

const SIGMOID: &str = "1.0 / (1.0 + exp(-x))";

const SIGMOID_PRIME: &str = "x * (1.0 - x)";