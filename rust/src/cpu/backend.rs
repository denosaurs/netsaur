use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::{ActivationCPULayer, BackendConfig, CPUCost, CPULayer, Dataset, DenseCPULayer, Layer, Conv2DCPULayer, Pool2DCPULayer};

pub struct CPUBackend {
    pub layers: Vec<CPULayer>,
    pub size: Vec<usize>,
    pub cost: CPUCost,
}

impl CPUBackend {
    pub fn new(config: BackendConfig) -> Self {
        let mut layers = Vec::new();
        let mut size = config.size.clone();
        for layer in config.layers {
            match layer {
                Layer::Activation(config) => {
                    let layer = ActivationCPULayer::new(config, IxDyn(&size));
                    layers.push(CPULayer::Activation(layer));
                }
                Layer::Conv2D(config) => {
                    let layer = Conv2DCPULayer::new(config, IxDyn(&size));
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Conv2D(layer));
                }
                Layer::Dropout1D(_config) => {
                    unimplemented!("Dropout1D is not implemented yet")
                }
                Layer::Dropout2D(_config) => {
                    unimplemented!("Dropout2D is not implemented yet")
                }
                Layer::Dense(config) => {
                    let layer = DenseCPULayer::new(config, IxDyn(&size));
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Dense(layer));
                }
                Layer::Flatten(_config) => {
                    unimplemented!("Flatten is not implemented yet")
                }
                Layer::Pool2D(config) => {
                    let layer = Pool2DCPULayer::new(config, IxDyn(&size));
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Pool2D(layer));
                }
            }
        }
        let cost = CPUCost::from(config.cost);
        Self { layers, cost, size }
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
