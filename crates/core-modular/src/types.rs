use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendConfig {
    pub silent: Option<bool>,
    pub size: Vec<usize>,
    pub layers: Vec<Layer>,
    pub cost: Cost,
    pub optimizer: Optimizer,
    pub scheduler: Scheduler,
    pub tolerance: Option<f32>,
    pub patience: Option<usize>,
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
    BatchNorm1D(BatchNormLayer),
    BatchNorm2D(BatchNormLayer),
    Conv2D(Conv2DLayer),
    ConvTranspose2D(ConvTranspose2DLayer),
    Pool2D(Pool2DLayer),
    Embedding(EmbeddingLayer),
    Flatten,
    LSTM(LSTMLayer),
    Dropout1D(DropoutLayer),
    Dropout2D(DropoutLayer),
    Softmax(SoftmaxLayer),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Elu,
    LeakyRelu,
    Linear,
    Relu,
    Relu6,
    Selu,
    Gelu,
    Sigmoid,
    Tanh,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JSTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DenseLayer {
    pub size: usize,
    pub init: Option<Init>,
    pub c: Option<f32>,
    pub l1_ratio: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Conv2DLayer {
    pub init: Option<Init>,
    pub kernel: Option<JSTensor>,
    pub kernel_size: Vec<usize>,
    pub padding: Option<Vec<usize>>,
    pub strides: Option<Vec<usize>>,
    pub c: Option<f32>,
    pub l1_ratio: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ConvTranspose2DLayer {
    pub init: Option<Init>,
    pub kernel: Option<JSTensor>,
    pub kernel_size: Vec<usize>,
    pub padding: Option<Vec<usize>>,
    pub strides: Option<Vec<usize>>,
    pub c: Option<f32>,
    pub l1_ratio: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Pool2DLayer {
    pub mode: usize, // 0 = avg, 1 = max
    pub strides: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingLayer {
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub c: Option<f32>,
    pub l1_ratio: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LSTMLayer {
    pub size: usize,
    pub init: Option<Init>,
    pub c: Option<f32>,
    pub l1_ratio: Option<f32>,
    pub return_sequences: Option<bool>,
    pub recurrent_activation: Option<Activation>,
    pub activation: Option<Activation>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DropoutLayer {
    pub probability: f32,
    pub inplace: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SoftmaxLayer {
    pub temperature: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchNormLayer {
    pub momentum: f32,
    pub epsilon: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActivationLayer {
    pub activation: Activation,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Cost {
    BinCrossEntropy,
    CrossEntropy,
    Hinge,
    Huber,
    MAE,
    MSE,
    SmoothHinge,
    Tukey,
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
#[serde(rename_all = "lowercase")]
pub struct AdamOptimizer {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub struct NadamOptimizer {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RMSPropOptimizer {
    pub decay_rate: f32,
    pub epsilon: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum Optimizer {
    SGD,
    Adam(AdamOptimizer),
    Nadam(NadamOptimizer),
    RMSProp(RMSPropOptimizer),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub struct DecayScheduler {
    pub rate: f32,
    pub step_size: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub struct OneCycleScheduler {
    pub max_rate: f32,
    pub step_size: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum Scheduler {
    None,
    LinearDecay(DecayScheduler),
    ExponentialDecay(DecayScheduler),
    OneCycle(OneCycleScheduler),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StepFunctionConfig {
    pub thresholds: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum PostProcessor {
    None,
    Sign,
    Step(StepFunctionConfig),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TrainOptions {
    pub datasets: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub epochs: usize,
    pub batches: usize,
    pub rate: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PredictOptions {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub layers: Option<Vec<usize>>,
    pub post_process: PostProcessor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RegularizeOptions {
    pub c: f32,
    pub l1_ratio: f32,
}
