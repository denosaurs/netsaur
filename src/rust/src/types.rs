use ndarray::ArrayD;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct BackendConfig {
    pub size: Vec<usize>,
    pub layers: Vec<Layer>,
    pub cost: Cost,
}

#[derive(Debug)]
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
#[serde(rename_all = "lowercase")] 
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

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")] 
pub struct TrainOptions {
    pub datasets: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub epochs: usize,
    pub rate: f32
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")] 
pub struct PredictOptions {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}