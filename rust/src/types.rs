use ndarray::ArrayD;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
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

#[derive(Deserialize, Debug, Clone)]
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
}

#[derive(Deserialize, Debug, Clone)]
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

#[derive(Deserialize, Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>
}

#[derive(Deserialize, Debug, Clone)]
pub struct DenseLayer {
    pub size: Vec<usize>,
    pub activation: Option<Activation>,
    pub init: Option<Init>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Conv2DLayer {
    pub init: Option<Init>,
    pub activation: Option<Activation>,
    pub kernel: Option<Tensor>,
    pub kernel_size: Vec<usize>,
    pub padding: usize,
    pub strides: Option<Vec<usize>>
}

#[derive(Deserialize, Debug, Clone)]
pub struct Pool2DLayer {
    pub mode: usize, // 0 = avg, 1 = max
    pub strides: Option<Vec<usize>>
}

#[derive(Deserialize, Debug, Clone)]
pub struct FlattenLayer {
    pub size: Vec<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Dropout1DLayer {
    pub probability: f32,
    pub inplace: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Dropout2DLayer {
    pub probability: f32,
    pub inplace: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ActivationLayer {
    pub activation: Activation,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Cost {
    MSE,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Init {
    Uniform,
    Xavier,
    XavierN,
    Kaiming,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TrainOptions {
    pub datasets: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub epochs: usize,
    pub rate: f32,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PredictOptions {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}
