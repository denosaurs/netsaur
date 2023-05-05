use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::{ActivationCPULayer, BackendConfig, CPUCost, CPULayer, Dataset, DenseCPULayer, Layer};

pub struct CPUBackend {
    pub layers: Vec<CPULayer>,
    pub cost: CPUCost,
}

impl CPUBackend {
    pub fn new(config: BackendConfig) -> Self {
        let mut layers = Vec::new();
        let mut size = IxDyn(&config.size);
        let batches = size[0];
        for layer in config.layers {
            match layer {
                Layer::Dense(layer) => {
                    let layer_size = layer.size.clone();
                    layers.push(CPULayer::Dense(DenseCPULayer::new(layer, size.clone())));
                    size = IxDyn(&[batches, layer_size[0]]);
                }
                Layer::Conv2D(_layer) => {
                    unimplemented!("Conv2D is not implemented yet")
                }
                Layer::Pool2D(_layer) => {
                    unimplemented!("Pool2D is not implemented yet")
                }
                Layer::Flatten(_layer) => {
                    unimplemented!("Flatten is not implemented yet")
                }
                Layer::Activation(layer) => {
                    layers.push(CPULayer::Activation(ActivationCPULayer::new(
                        layer,
                        size.clone(),
                    )));
                }
            }
        }
        let cost = CPUCost::from(config.cost);
        Self { layers, cost }
    }

    pub fn forward_propagate(&mut self, mut inputs: ArrayD<f32>) -> ArrayD<f32> {
        for layer in &mut self.layers {
            inputs = layer.forward_propagate(inputs);
        }
        inputs
    }

    pub fn backward_propagate<'b>(
        &mut self,
        outputs: ArrayViewD<'b, f32>,
        data: ArrayViewD<'b, f32>,
        rate: f32,
    ) -> ArrayD<f32> {
        let mut d_outputs = (self.cost.prime)(outputs, data);
        for layer in self.layers.iter_mut().rev() {
            d_outputs = layer.backward_propagate(d_outputs, rate);
        }
        d_outputs
    }

    pub fn train(&mut self, datasets: Vec<Dataset>, epochs: usize, rate: f32) {
        let mut epoch = 0;
        while epoch < epochs {
            for dataset in &datasets {
                let outputs = self.forward_propagate(dataset.inputs.clone());
                self.backward_propagate(outputs.view(), dataset.outputs.view(), rate);
            }

            epoch += 1
        }
    }

    pub fn predict(&mut self, data: ArrayD<f32>) -> ArrayD<f32> {
        for layer in &mut self.layers {
            layer.reset(1)
        }
        self.forward_propagate(data)
    }
}
