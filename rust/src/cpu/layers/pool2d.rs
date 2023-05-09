use ndarray::{s, Array4, ArrayD, Ix4, IxDyn};

use crate::Pool2DLayer;

pub struct Pool2DCPULayer {
    pub strides: Vec<usize>,
    pub inputs: Array4<f32>,
    pub outputs: Array4<f32>,
}

impl Pool2DCPULayer {
    pub fn new(config: Pool2DLayer, size: IxDyn) -> Self {
        let strides = config.strides.unwrap_or(vec![1, 1]);
        let input_size = Ix4(size[0], size[1], size[2], size[3]);
        let output_y = size[2] / strides[0];
        let output_x = size[3] / strides[1];
        let output_size = Ix4(size[0], size[1], output_y, output_x);

        Self {
            strides,
            inputs: Array4::zeros(input_size),
            outputs: Array4::zeros(output_size),
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.outputs.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let input_size = self.inputs.shape();
        self.inputs = Array4::zeros((batches, input_size[1], input_size[2], input_size[3]));
        let output_size = self.outputs.shape();
        self.outputs = Array4::zeros((batches, output_size[1], output_size[2], output_size[3]));
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix4>().unwrap();

        let (batches, channels, output_y, output_x) = self.outputs.dim();

        for b in 0..batches {
            for c in 0..channels {
                for y in 0..output_y {
                    for x in 0..output_x {
                        let input_y = y * self.strides[0];
                        let input_x = x * self.strides[1];
                        let stride_y = (y + 1) * self.strides[0];
                        let stride_x = (x + 1) * self.strides[1];
                        self.outputs[[b, c, y, x]] = self
                            .inputs
                            .slice(s![b, c, input_y..stride_y, input_x..stride_x])
                            .fold(0.0, |max_value, value| {
                                if value > &max_value {
                                    return value.clone();
                                }
                                max_value
                            })
                    }
                }
            }
        }

        self.outputs.clone().into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>, _rate: f32) -> ArrayD<f32> {
        d_outputs
    }
}
