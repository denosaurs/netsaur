use ndarray::{s, Array4, Array5, ArrayD, Ix4, Ix5, IxDyn};

use crate::Pool2DLayer;

pub struct Pool2DGPULayer {
    pub strides: Vec<usize>,
    pub inputs: Array4<f32>,
    pub indices: Array5<usize>,
    pub outputs: Array4<f32>,
    pub max: bool,
}

impl Pool2DGPULayer {
    pub fn new(config: Pool2DLayer, size: IxDyn) -> Self {
        let strides = config.strides.unwrap_or(vec![1, 1]);
        let input_size = Ix4(size[0], size[1], size[2], size[3]);
        let output_y = size[2] / strides[0];
        let output_x = size[3] / strides[1];
        let indice_size = Ix5(size[0], size[1], output_y, output_x, 2);
        let output_size = Ix4(size[0], size[1], output_y, output_x);
        let max = config.mode == 1;
        Self {
            strides,
            inputs: Array4::zeros(input_size),
            indices: Array5::zeros(indice_size),
            outputs: Array4::zeros(output_size),
            max,
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.outputs.shape().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        let input_size = self.inputs.shape();
        self.inputs = Array4::zeros((batches, input_size[1], input_size[2], input_size[3]));
        let indice_size = self.outputs.shape();
        self.indices = Array5::zeros((batches, indice_size[1], indice_size[2], indice_size[3], 2));
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
                        if self.max {
                            let mut max_index = (0, 0);
                            let mut max_value = 0.0;
                            self.inputs
                                .slice(s![b, c, input_y..stride_y, input_x..stride_x])
                                .indexed_iter()
                                .for_each(|(index, value)| {
                                    if value > &max_value {
                                        max_value = *value;
                                        max_index = index;
                                    }
                                });
                            let mut position = self.indices.slice_mut(s![b, c, y, x, ..]);
                            position[0] = max_index.0.into();
                            position[1] = max_index.1.into();
                            self.outputs[[b, c, y, x]] = max_value;
                        } else {
                            self.outputs[[b, c, y, x]] = self
                                .inputs
                                .slice(s![b, c, input_y..stride_y, input_x..stride_x])
                                .mean()
                                .unwrap();
                        }
                    }
                }
            }
        }

        self.outputs.clone().into_dyn()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        let d_outputs = d_outputs.into_dimensionality::<Ix4>().unwrap();

        let (batches, channels, output_y, output_x) = self.outputs.dim();

        let mut d_inputs = Array4::zeros(self.inputs.dim());
        for b in 0..batches {
            for c in 0..channels {
                for y in 0..output_y {
                    for x in 0..output_x {
                        let input_y = y * self.strides[0];
                        let input_x = x * self.strides[1];
                        let stride_y = (y + 1) * self.strides[0];
                        let stride_x = (x + 1) * self.strides[1];
                        if self.max {
                            let index = self.indices.slice(s![b, c, y, x, ..]);
                            d_inputs[[
                                b,
                                c,
                                input_y + index[0] as usize,
                                input_x + index[1] as usize,
                            ]] = d_outputs[[b, c, y, x]];
                        } else {
                            d_inputs
                                .slice_mut(s![b, c, input_y..stride_y, input_x..stride_x])
                                .fill(
                                    d_outputs[[b, c, y, x]]
                                        / self.strides[0] as f32
                                        / self.strides[1] as f32,
                                );
                        }
                    }
                }
            }
        }

        d_inputs.into_dyn()
    }
}
