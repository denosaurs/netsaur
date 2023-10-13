mod activation;
mod batchnorm1d;
mod batchnorm2d;
mod conv2d;
mod convtrans2d;
mod dense;
mod dropout;
mod flatten;
mod pool2d;

pub use activation::*;
pub use batchnorm1d::*;
pub use batchnorm2d::*;
pub use conv2d::*;
pub use convtrans2d::*;
pub use dense::*;
pub use dropout::*;
pub use flatten::*;
pub use pool2d::*;

use ndarray::ArrayD;

pub enum GPULayer {
    Activation(ActivationGPULayer),
    Conv2D(Conv2DGPULayer),
    ConvTranspose2D(ConvTranspose2DGPULayer),
    Dense(DenseGPULayer),
    Dropout1D(Dropout1DGPULayer),
    Dropout2D(Dropout2DGPULayer),
    Flatten(FlattenGPULayer),
    Pool2D(Pool2DGPULayer),
    Softmax(SoftmaxGPULayer),
    BatchNorm1D(BatchNorm1DGPULayer),
    BatchNorm2D(BatchNorm2DGPULayer),
}

impl GPULayer {
    pub fn output_size(&mut self) -> Vec<usize> {
        match self {
            GPULayer::Activation(layer) => layer.output_size(),
            GPULayer::BatchNorm1D(layer) => layer.output_size(),
            GPULayer::BatchNorm2D(layer) => layer.output_size(),
            GPULayer::Conv2D(layer) => layer.output_size(),
            GPULayer::ConvTranspose2D(layer) => layer.output_size(),
            GPULayer::Dense(layer) => layer.output_size(),
            GPULayer::Dropout1D(layer) => layer.output_size(),
            GPULayer::Dropout2D(layer) => layer.output_size(),
            GPULayer::Flatten(layer) => layer.output_size(),
            GPULayer::Pool2D(layer) => layer.output_size(),
            GPULayer::Softmax(layer) => layer.output_size(),
        }
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>, training: bool) -> ArrayD<f32> {
        match self {
            GPULayer::Activation(layer) => layer.forward_propagate(inputs),
            GPULayer::BatchNorm1D(layer) => layer.forward_propagate(inputs, training),
            GPULayer::BatchNorm2D(layer) => layer.forward_propagate(inputs, training),
            GPULayer::Conv2D(layer) => layer.forward_propagate(inputs),
            GPULayer::ConvTranspose2D(layer) => layer.forward_propagate(inputs),
            GPULayer::Dense(layer) => layer.forward_propagate(inputs),
            GPULayer::Dropout1D(layer) => layer.forward_propagate(inputs, training),
            GPULayer::Dropout2D(layer) => layer.forward_propagate(inputs, training),
            GPULayer::Flatten(layer) => layer.forward_propagate(inputs),
            GPULayer::Pool2D(layer) => layer.forward_propagate(inputs),
            GPULayer::Softmax(layer) => layer.forward_propagate(inputs),
        }
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            GPULayer::Activation(layer) => layer.backward_propagate(d_outputs),
            GPULayer::BatchNorm1D(layer) => layer.backward_propagate(d_outputs),
            GPULayer::BatchNorm2D(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Conv2D(layer) => layer.backward_propagate(d_outputs),
            GPULayer::ConvTranspose2D(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Dense(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Dropout1D(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Dropout2D(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Flatten(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Pool2D(layer) => layer.backward_propagate(d_outputs),
            GPULayer::Softmax(layer) => layer.backward_propagate(d_outputs),
        }
    }

    pub fn reset(&mut self, batches: usize) {
        match self {
            GPULayer::Activation(layer) => layer.reset(batches),
            GPULayer::BatchNorm1D(layer) => layer.reset(batches),
            GPULayer::BatchNorm2D(layer) => layer.reset(batches),
            GPULayer::Conv2D(layer) => layer.reset(batches),
            GPULayer::Dense(layer) => layer.reset(batches),
            GPULayer::Dropout1D(layer) => layer.reset(batches),
            GPULayer::Dropout2D(layer) => layer.reset(batches),
            GPULayer::Flatten(layer) => layer.reset(batches),
            GPULayer::Pool2D(layer) => layer.reset(batches),
            GPULayer::Softmax(layer) => layer.reset(batches),
            GPULayer::ConvTranspose2D(layer) => layer.reset(batches),
        }
    }
}
