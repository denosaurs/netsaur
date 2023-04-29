use ndarray::ArrayD;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct BackendConfig {
    pub size: Vec<usize>,
    pub layers: Vec<Layer>,
    pub cost: Cost,
}

pub struct Dataset {
    pub inputs: ArrayD<f32>,
    pub outputs: ArrayD<f32>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")] 
pub enum Layer {
    Activation(Activation),
    Dense(Dense),
}

#[derive(Deserialize, Debug)]
pub enum Activation {
    Sigmoid,
    Tanh,
}

#[derive(Deserialize, Debug)]
pub struct Dense {
    pub size: Vec<usize>,
    pub activation: Option<Activation>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")] 
pub enum Cost {
    MSE,
}