use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendConfig {
    pub silent: Option<bool>,
    pub size: Vec<usize>,
    pub layers: Vec<Layer>,
    pub cost: Cost,
}

#[derive(Debug)]
pub struct Dataset {
    pub inputs: ArrayD<f32>,
    pub outputs: ArrayD<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum Layer {
    Activation(ActivationLayer),
    Dense(DenseLayer),
    Conv2D(Conv2DLayer),
    Pool2D(Pool2DLayer),
    Flatten(FlattenLayer),
    Dropout1D(Dropout1DLayer),
    Dropout2D(Dropout2DLayer),
    Softmax,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Sigmoid,
    Tanh,
    Linear,
    Relu,
    Relu6,
    LeakyRelu,
    Elu,
    Selu,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JSTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DenseLayer {
    pub size: Vec<usize>,
    pub init: Option<Init>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Conv2DLayer {
    pub init: Option<Init>,
    pub kernel: Option<JSTensor>,
    pub kernel_size: Vec<usize>,
    pub padding: Option<Vec<usize>>,
    pub strides: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Pool2DLayer {
    pub mode: usize, // 0 = avg, 1 = max
    pub strides: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlattenLayer {
    pub size: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dropout1DLayer {
    pub probability: f32,
    pub inplace: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dropout2DLayer {
    pub probability: f32,
    pub inplace: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActivationLayer {
    pub activation: Activation,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Cost {
    CrossEntropy,
    MSE,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Init {
    Uniform,
    Xavier,
    XavierN,
    Kaiming,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TrainOptions {
    pub datasets: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub epochs: usize,
    pub rate: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PredictOptions {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}