use ndarray::ArrayD;
use ndarray_rand::{
    rand_distr::{Normal, Uniform},
    RandomExt,
};

use crate::{length, Init};

pub struct CPUInit {
    pub init: fn(size: &[usize], inputs: &[usize], outputs: &[usize]) -> ArrayD<f32>,
}

impl CPUInit {
    pub fn from(init: Init) -> Self {
        match init {
            Init::Uniform => CPUInit { init: uniform },
            Init::Xavier => CPUInit { init: xavier },
            Init::XavierN => CPUInit { init: xaviern },
            Init::Kaiming => CPUInit { init: kaiming },
        }
    }

    pub fn from_default(init: Option<Init>, default: Init) -> Self {
        if let Some(init) = init {
            return CPUInit::from(init)
        }
        CPUInit::from(default)
    }
}

pub fn uniform(size: &[usize], _inputs: &[usize], _outputs: &[usize]) -> ArrayD<f32> {
    ArrayD::random(size.clone(), Uniform::new(-1.0, 1.0))
}

pub fn xavier(size: &[usize], inputs: &[usize], _outputs: &[usize]) -> ArrayD<f32> {
    let bounds = 1.0 / (length(inputs.to_vec()) as f32).sqrt();
    ArrayD::random(size.clone(), Uniform::new(-bounds, bounds))
}

pub fn xaviern(size: &[usize], inputs: &[usize], outputs: &[usize]) -> ArrayD<f32> {
    let bounds =
        (6.0 as f32).sqrt() / ((length(inputs.to_vec()) + length(outputs.to_vec())) as f32).sqrt();
    ArrayD::random(size.clone(), Uniform::new(-bounds, bounds))
}

pub fn kaiming(size: &[usize], inputs: &[usize], _outputs: &[usize]) -> ArrayD<f32> {
    let deviation = (2.0 / (length(inputs.to_vec()) as f32)).sqrt();
    ArrayD::random(size.clone(), Normal::new(0.0, deviation).unwrap())
}
