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
    Activation(ActivationLayer),
    Dense(DenseLayer),
    Conv2D(Conv2DLayer),
    Pool2D(Pool2DLayer),
    Flatten(FlattenLayer),
    Dropout1D(Dropout1DLayer),
    Dropout2D(Dropout2DLayer),
}

#[derive(Deserialize, Debug)]
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

#[derive(Deserialize, Debug)]
pub struct DenseLayer {
    pub size: Vec<usize>,
    pub activation: Option<Activation>,
    pub init: Option<Init>,
}

#[derive(Deserialize, Debug)]
pub struct Conv2DLayer {
    pub init: Option<Init>,
    pub activation: Option<Activation>,
    pub kernel: Option<Vec<f32>>,
    pub kernel_size: Vec<usize>,
    pub padding: u32,
    pub strides: Option<Vec<u32>>
}

#[derive(Deserialize, Debug)]
pub struct Pool2DLayer {
    pub mode: usize, // 0 = avg, 1 = max
    pub strides: Option<Vec<u32>>
}

#[derive(Deserialize, Debug)]
pub struct FlattenLayer {
    pub size: Vec<usize>,
}

#[derive(Deserialize, Debug)]
pub struct Dropout1DLayer {
    pub probability: f32,
    pub inplace: bool,
}

#[derive(Deserialize, Debug)]
pub struct Dropout2DLayer {
    pub probability: f32,
    pub inplace: bool,
}

#[derive(Deserialize, Debug)]
pub struct ActivationLayer {
    pub activation: Activation,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Cost {
    MSE,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Init {
    Uniform,
    Xavier,
    XavierN,
    Kaiming,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TrainOptions {
    pub datasets: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub epochs: usize,
    pub rate: f32,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PredictOptions {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}
