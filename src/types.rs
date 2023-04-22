use ndarray::ArrayD;

use crate::CPUBackend;

pub struct Dataset {
    pub inputs: ArrayD<f32>,
    pub outputs: ArrayD<f32>,
}

pub enum Layer {
    Activation(Activation),
    Dense(Dense),
}

#[derive(Clone)]
pub enum Activation {
    Sigmoid,
    Tanh,
}

#[derive(Clone)]
pub struct Dense {
    pub size: usize,
    pub activation: Option<Activation>,
}

pub enum Cost {
    MSE,
}

pub enum BackendType {
    CPU,
}

pub enum Backend {
    CPU(CPUBackend),
}
