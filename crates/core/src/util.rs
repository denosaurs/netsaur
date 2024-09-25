use std::slice::from_raw_parts;

use ndarray::ArrayD;
use safetensors::tensor::TensorView;
use serde::Deserialize;

#[derive(Clone)]
pub struct Logger {
    pub log: fn(string: String) -> (),
}

#[derive(Clone)]
pub struct Timer {
    pub now: fn() -> u128,
}

pub fn length(shape: Vec<usize>) -> usize {
    return shape.iter().fold(1, |i, x| i * x);
}

pub fn decode_array(ptr: *const f32, shape: Vec<usize>) -> ArrayD<f32> {
    let buffer = unsafe { from_raw_parts(ptr, length(shape.clone())) };
    let vec = Vec::from(buffer);
    return ArrayD::from_shape_vec(shape, vec).unwrap();
}

pub fn decode_json<'a, T>(ptr: *const u8, len: usize) -> T
where
    T: Deserialize<'a>,
{
    let buffer = unsafe { from_raw_parts(ptr, len) };
    let json = std::str::from_utf8(&buffer[0..len]).unwrap();
    return serde_json::from_str(&json).unwrap();
}

pub fn to_arr(view: TensorView) -> ArrayD<f32> {
    let slice: &[f32] =
        unsafe { from_raw_parts(view.data().as_ptr() as *const f32, view.data().len() / 4) };
    return ArrayD::from_shape_vec(view.shape(), slice.to_vec()).unwrap();
}
