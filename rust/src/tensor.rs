use std::{borrow::Cow, slice::from_raw_parts};

use ndarray::ArrayViewD;
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
