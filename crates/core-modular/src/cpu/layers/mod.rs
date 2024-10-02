mod activation;
mod batchnorm1d;
mod batchnorm2d;
mod conv2d;
mod convtrans2d;
mod dense;
mod dropout;
mod embedding;
mod flatten;
mod lstm;
mod pool2d;

pub use activation::*;
pub use batchnorm1d::*;
pub use batchnorm2d::*;
pub use conv2d::*;
pub use convtrans2d::*;
pub use dense::*;
pub use dropout::*;
pub use embedding::*;
pub use flatten::*;
pub use lstm::*;
pub use pool2d::*;

use ndarray::{ArrayD, ArrayViewMutD};

#[repr(C)]
pub enum CPULayer {
    Activation(ActivationCPULayer),
    Conv2D(Conv2DCPULayer),
    ConvTranspose2D(ConvTranspose2DCPULayer),
    Dense(DenseCPULayer),
    Dropout1D(Dropout1DCPULayer),
    Dropout2D(Dropout2DCPULayer),
    Flatten(FlattenCPULayer),
    Embedding(EmbeddingCPULayer),
    LSTM(LSTMCPULayer),
    Pool2D(Pool2DCPULayer),
    Softmax(SoftmaxCPULayer),
    BatchNorm1D(BatchNorm1DCPULayer),
    BatchNorm2D(BatchNorm2DCPULayer),
}

impl CPULayer {
    pub fn output_size(&mut self) -> Vec<usize> {
        match self {
            CPULayer::Activation(layer) => layer.output_size(),
            CPULayer::BatchNorm1D(layer) => layer.output_size(),
            CPULayer::BatchNorm2D(layer) => layer.output_size(),
            CPULayer::Conv2D(layer) => layer.output_size(),
            CPULayer::ConvTranspose2D(layer) => layer.output_size(),
            CPULayer::Dense(layer) => layer.output_size(),
            CPULayer::Dropout1D(layer) => layer.output_size(),
            CPULayer::Dropout2D(layer) => layer.output_size(),
            CPULayer::Embedding(layer) => layer.output_size(),
            CPULayer::LSTM(layer) => layer.output_size(),
            CPULayer::Flatten(layer) => layer.output_size(),
            CPULayer::Pool2D(layer) => layer.output_size(),
            CPULayer::Softmax(layer) => layer.output_size(),
        }
    }

    pub fn get_learnable_params(&mut self) -> Vec<ArrayViewMutD<f32>> {
        match self {
            CPULayer::BatchNorm1D(layer) => {
                vec![
                    layer.beta.view_mut().into_dyn(),
                    layer.gamma.view_mut().into_dyn(),
                ]
            }
            CPULayer::BatchNorm2D(layer) => {
                vec![
                    layer.beta.view_mut().into_dyn(),
                    layer.gamma.view_mut().into_dyn(),
                ]
            }
            CPULayer::Conv2D(layer) => vec![
                layer.weights.view_mut().into_dyn(),
                layer.biases.view_mut().into_dyn(),
            ],
            CPULayer::ConvTranspose2D(layer) => {
                vec![
                    layer.weights.view_mut().into_dyn(),
                    layer.biases.view_mut().into_dyn(),
                ]
            }
            CPULayer::Dense(layer) => vec![
                layer.weights.view_mut().into_dyn(),
                layer.biases.view_mut().into_dyn(),
            ],
            CPULayer::Embedding(layer) => vec![layer.embeddings.view_mut().into_dyn()],
            CPULayer::LSTM(layer) => vec![
                layer.w_ih.view_mut().into_dyn(),
                layer.w_hh.view_mut().into_dyn(),
                layer.biases.view_mut().into_dyn(),
            ],
            _ => vec![],
        }
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>, training: bool) -> ArrayD<f32> {
        match self {
            CPULayer::Activation(layer) => layer.forward_propagate(inputs),
            CPULayer::BatchNorm1D(layer) => layer.forward_propagate(inputs, training),
            CPULayer::BatchNorm2D(layer) => layer.forward_propagate(inputs, training),
            CPULayer::Conv2D(layer) => layer.forward_propagate(inputs),
            CPULayer::ConvTranspose2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Dense(layer) => layer.forward_propagate(inputs),
            CPULayer::Dropout1D(layer) => layer.forward_propagate(inputs, training),
            CPULayer::Dropout2D(layer) => layer.forward_propagate(inputs, training),
            CPULayer::Embedding(layer) => layer.forward_propagate(inputs),
            CPULayer::LSTM(layer) => layer.forward_propagate(inputs),
            CPULayer::Flatten(layer) => layer.forward_propagate(inputs),
            CPULayer::Pool2D(layer) => layer.forward_propagate(inputs),
            CPULayer::Softmax(layer) => layer.forward_propagate(inputs),
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            CPULayer::Activation(layer) => layer.backward_propagate(d_outputs),
            CPULayer::BatchNorm1D(layer) => layer.backward_propagate(d_outputs),
            CPULayer::BatchNorm2D(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Conv2D(layer) => layer.backward_propagate(d_outputs),
            CPULayer::ConvTranspose2D(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Dense(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Dropout1D(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Dropout2D(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Embedding(layer) => layer.backward_propagate(d_outputs),
            CPULayer::LSTM(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Flatten(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Pool2D(layer) => layer.backward_propagate(d_outputs),
            CPULayer::Softmax(layer) => layer.backward_propagate(d_outputs),
        }
    }

    pub fn reset(&mut self, batches: usize) {
        match self {
            CPULayer::Activation(layer) => layer.reset(batches),
            CPULayer::BatchNorm1D(layer) => layer.reset(batches),
            CPULayer::BatchNorm2D(layer) => layer.reset(batches),
            CPULayer::Conv2D(layer) => layer.reset(batches),
            CPULayer::Dense(layer) => layer.reset(batches),
            CPULayer::Dropout1D(layer) => layer.reset(batches),
            CPULayer::Dropout2D(layer) => layer.reset(batches),
            CPULayer::Embedding(layer) => layer.reset(batches),
            CPULayer::LSTM(layer) => layer.reset(batches),
            CPULayer::Flatten(layer) => layer.reset(batches),
            CPULayer::Pool2D(layer) => layer.reset(batches),
            CPULayer::Softmax(layer) => layer.reset(batches),
            CPULayer::ConvTranspose2D(layer) => layer.reset(batches),
        }
    }
}
