extern crate ndarray;
use ndarray::{Array1, ArrayD, Axis};

pub struct LayerNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(hidden_size: usize, epsilon: f32) -> Self {
        LayerNorm {
            gamma: Array1::ones(hidden_size),
            beta: Array1::zeros(hidden_size),
            epsilon,
        }
    }

    pub fn forward(&self, input: ArrayD<f32>) -> ArrayD<f32> {
        let shape = input.shape();
        let last_axis = shape.len() - 1;

        let mean = input.mean_axis(Axis(last_axis)).unwrap();
        let variance = input.var_axis(Axis(last_axis), 0.0);

        let mut normalized_input = input.clone();
        normalized_input
            .axis_iter_mut(Axis(last_axis))
            .enumerate()
            .for_each(|(i, mut row)| {
                let mean_i = mean[i];
                let var_i = variance[i].sqrt() + self.epsilon;
                row -= mean_i;
                row /= var_i;
            });

        normalized_input
            .axis_iter_mut(Axis(last_axis))
            .for_each(|mut item| {
                let new = &item * &self.gamma + &self.beta;
                item.assign(&new);
            });
        normalized_input
    }
}
