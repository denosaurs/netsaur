use std::collections::HashMap;

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use safetensors::{serialize, SafeTensors};

use crate::{
    to_arr, ActivationCPULayer, BackendConfig, CPUCost, CPULayer, Conv2DCPULayer, Dataset,
    DenseCPULayer, Dropout1DCPULayer, FlattenCPULayer, Layer, Logger, Pool2DCPULayer,
    SoftmaxCPULayer, Tensor, Dropout2DCPULayer,
};

pub struct CPUBackend {
    pub silent: bool,
    pub config: BackendConfig,
    pub layers: Vec<CPULayer>,
    pub size: Vec<usize>,
    pub cost: CPUCost,
    pub logger: Logger,
}

impl CPUBackend {
    pub fn new(config: BackendConfig, logger: Logger, tensors: Option<SafeTensors>) -> Self {
        let mut layers = Vec::new();
        let mut size = config.size.clone();
        for (i, layer) in config.layers.iter().enumerate() {
            match layer.clone() {
                Layer::Activation(config) => {
                    let layer = ActivationCPULayer::new(config, IxDyn(&size));
                    layers.push(CPULayer::Activation(layer));
                }
                Layer::Conv2D(config) => {
                    let layer = if let Some(tensors) = &tensors {
                        let weights = to_arr(tensors.tensor(&format!("{}w", i)).unwrap());
                        let biases = to_arr(tensors.tensor(&format!("{}b", i)).unwrap());
                        Conv2DCPULayer::new(config, IxDyn(&size), Some(weights), Some(biases))
                    } else {
                        Conv2DCPULayer::new(config, IxDyn(&size), None, None)
                    };
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Conv2D(layer));
                }
                Layer::Dropout1D(config) => {
                    let layer = Dropout1DCPULayer::new(config, IxDyn(&size));
                    layers.push(CPULayer::Dropout1D(layer));
                }
                Layer::Dropout2D(config) => {
                    let layer = Dropout2DCPULayer::new(config, IxDyn(&size));
                    layers.push(CPULayer::Dropout2D(layer));
                }
                Layer::Dense(config) => {
                    let layer = if let Some(tensors) = &tensors {
                        let weights = to_arr(tensors.tensor(&format!("{}w", i)).unwrap());
                        let biases = to_arr(tensors.tensor(&format!("{}b", i)).unwrap());
                        DenseCPULayer::new(config, IxDyn(&size), Some(weights), Some(biases))
                    } else {
                        DenseCPULayer::new(config, IxDyn(&size), None, None)
                    };
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Dense(layer));
                }
                Layer::Flatten(config) => {
                    let layer = FlattenCPULayer::new(config, IxDyn(&size));
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Flatten(layer));
                }
                Layer::Pool2D(config) => {
                    let layer = Pool2DCPULayer::new(config, IxDyn(&size));
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Pool2D(layer));
                }
                Layer::Softmax => {
                    let layer = SoftmaxCPULayer::new(IxDyn(&size));
                    layers.push(CPULayer::Softmax(layer));
                }
            }
        }
        let cost = CPUCost::from(config.cost.clone());
        let silent = config.silent.is_some();
        Self {
            logger,
            silent,
            config,
            layers,
            cost,
            size,
        }
    }

    pub fn forward_propagate(&mut self, mut inputs: ArrayD<f32>, training: bool) -> ArrayD<f32> {
        for layer in &mut self.layers {
            inputs = layer.forward_propagate(inputs, training);
        }
        inputs
    }

    pub fn backward_propagate<'b>(
        &mut self,
        outputs: ArrayViewD<'b, f32>,
        data: ArrayViewD<'b, f32>,
        rate: f32,
    ) -> ArrayD<f32> {
        let mut d_outputs = (self.cost.prime)(data, outputs);
        for layer in self.layers.iter_mut().rev() {
            d_outputs = layer.backward_propagate(d_outputs, rate);
        }
        d_outputs
    }

    pub fn train(&mut self, datasets: Vec<Dataset>, epochs: usize, batches: usize, rate: f32) {
        let mut epoch = 0;
        while epoch < epochs {
            let mut total = 0.0;
            for (i, dataset) in datasets.iter().enumerate() {
                let outputs = self.forward_propagate(dataset.inputs.clone(), true);
                self.backward_propagate(outputs.view(), dataset.outputs.view(), rate);
                total += (self.cost.cost)(outputs.view(), dataset.outputs.view());
                if !self.silent && i % batches == 0 {
                    let cost = total / batches as f32;
                    let msg = format!("Epoch={}, Dataset={}, Cost={}", epoch, i, cost);
                    (self.logger.log)(msg);
                    total = 0.0;
                }
            }

            epoch += 1
        }
    }

    pub fn predict(&mut self, data: ArrayD<f32>) -> ArrayD<f32> {
        for layer in &mut self.layers {
            layer.reset(1)
        }
        self.forward_propagate(data, false)
    }

    pub fn save(&self) -> Vec<u8> {
        let mut tensors = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                CPULayer::Conv2D(layer) => {
                    let weights = Tensor::new(layer.weights.view().into_dyn());
                    tensors.push((format!("{}w", i), weights));
                    let biases = Tensor::new(layer.biases.view().into_dyn());
                    tensors.push((format!("{}b", i), biases));
                }
                CPULayer::Dense(layer) => {
                    let weights = Tensor::new(layer.weights.view().into_dyn());
                    tensors.push((format!("{}w", i), weights));
                    let biases = Tensor::new(layer.biases.view().into_dyn());
                    tensors.push((format!("{}b", i), biases));
                }
                _ => {}
            }
        }
        let config = serde_json::to_string(&self.config).unwrap();
        let metadata = HashMap::from([("metadata".to_string(), config)]);
        serialize(tensors, &Some(metadata)).unwrap()
    }

    pub fn load(buffer: &[u8], logger: Logger) -> Self {
        let tensors = SafeTensors::deserialize(buffer).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(buffer).unwrap();
        let data = metadata.metadata().as_ref().unwrap();
        let json = data.get("metadata").unwrap();
        let config = serde_json::from_str(json).unwrap();

        CPUBackend::new(config, logger, Some(tensors))
    }
}
