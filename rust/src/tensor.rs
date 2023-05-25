use std::{borrow::Cow, slice::from_raw_parts};

use ndarray::{ArrayD, ArrayViewD};
use safetensors::{Dtype, View};

pub struct Tensor<'a> {
    pub data: ArrayViewD<'a, f32>,
}

impl<'a> Tensor<'a> {
    pub fn new(data: ArrayViewD<'a, f32>) -> Self {
        Self { data }
    }
}

impl<'a> View for Tensor<'a> {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn data(&self) -> Cow<[u8]> {
        let slice = self.data.as_slice().expect("Non contiguous tensors");
        let new_slice: &[u8] =
            unsafe { from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
        Cow::from(new_slice)
    }

    fn data_len(&self) -> usize {
        self.data.len() * 4
    }
}

#[derive(Debug)]
pub struct DenseTensors {
    pub weights: ArrayD<f32>,
    pub biases: ArrayD<f32>,
}

#[derive(Debug)]
pub struct ConvTensors {
    pub weights: ArrayD<f32>,
    pub biases: ArrayD<f32>,
}

#[derive(Debug)]
pub struct BatchNormTensors {
    pub gamma: ArrayD<f32>,
    pub beta: ArrayD<f32>,
    pub running_mean: ArrayD<f32>,
    pub running_var: ArrayD<f32>,
}

#[derive(Debug)]
pub enum Tensors {
    Dense(DenseTensors),
    Conv(ConvTensors),
    BatchNorm(BatchNormTensors),
}

pub trait GetTensor {
    fn get(&mut self) -> Option<Tensors>;
}

impl GetTensor for Option<Vec<Tensors>> {
    fn get(&mut self) -> Option<Tensors> {
        if let Some(tensors) = self {
            return Some(tensors.remove(0));
        }
        None
    }
}
