use ndarray::{ArrayD, IxDyn};
use ndarray_rand::{
    rand_distr::{Normal, Uniform},
    RandomExt,
};

use crate::Init;

pub struct GPUInit {
    pub init: Init,
}

impl GPUInit {
    pub fn from(init: Init) -> Self {
        Self { init }
    }

    pub fn init(&self, size: IxDyn, input_size: usize, output_size: usize) -> ArrayD<f32> {
        match self.init {
            Init::Uniform => uniform(size),
            Init::Xavier => xavier(size, input_size),
            Init::XavierN => xaviern(size, input_size, output_size),
            Init::Kaiming => kaiming(size, input_size),
        }
    }

    pub fn from_default(init: Option<Init>, default: Init) -> Self {
        if let Some(init) = init {
            return Self { init };
        }
        Self { init: default }
    }
}

pub fn uniform(size: IxDyn) -> ArrayD<f32> {
    ArrayD::random(size, Uniform::new(-1.0, 1.0))
}

pub fn xavier(size: IxDyn, input_size: usize) -> ArrayD<f32> {
    let bounds = 1.0 / (input_size as f32).sqrt();
    ArrayD::random(size, Uniform::new(-bounds, bounds))
}

pub fn xaviern(size: IxDyn, input_size: usize, output_size: usize) -> ArrayD<f32> {
    let bounds = (6.0 as f32).sqrt() / ((input_size + output_size) as f32).sqrt();
    ArrayD::random(size, Uniform::new(-bounds, bounds))
}

pub fn kaiming(size: IxDyn, input_size: usize) -> ArrayD<f32> {
    let deviation = (2.0 / (input_size as f32)).sqrt();
    ArrayD::random(size, Normal::new(0.0, deviation).unwrap())
}
