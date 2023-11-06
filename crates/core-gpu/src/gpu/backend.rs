use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn, Dimension};
use safetensors::{serialize, SafeTensors};

use crate::{
    to_arr, ActivationGPULayer, BackendConfig, Dataset, DenseGPULayer, DenseTensors, GPUCost,
    GPULayer, GPUScheduler, GetTensor, Layer, Logger, Tensor, Tensors, WGPUBackend, WGPUBuffer,
    WGPUDataset,
};

pub struct Backend {
    pub backend: WGPUBackend,
    pub silent: bool,
    pub config: BackendConfig,
    pub layers: Vec<GPULayer>,
    pub size: Vec<usize>,
    pub cost: GPUCost,
    pub scheduler: GPUScheduler,
    pub logger: Logger,
}

impl Backend {
    pub fn new(
        mut backend: WGPUBackend,
        config: BackendConfig,
        logger: Logger,
        mut tensors: Option<Vec<Tensors>>,
    ) -> Self {
        let mut layers = Vec::new();
        let mut size = IxDyn(&config.size);
        for layer in config.layers.iter() {
            match layer.clone() {
                Layer::Activation(config) => {
                    let layer = ActivationGPULayer::new(&mut backend, config, &mut size);
                    layers.push(GPULayer::Activation(layer));
                }
                Layer::Dense(config) => {
                    let layer = DenseGPULayer::new(&mut backend, config, &mut size, tensors.get());
                    layers.push(GPULayer::Dense(layer));
                }
                _ => unimplemented!(),
            };
        }

        Self {
            logger,
            layers,
            size: size.as_array_view().to_vec(),
            silent: config.silent.is_some_and(|x| x == true),
            cost: GPUCost::from(&mut backend, config.cost.clone(), size),
            scheduler: GPUScheduler::from(&config.scheduler),
            config,
            backend,
        }
    }

    pub fn forward_propagate<'a>(&'a mut self, mut inputs: &'a WGPUBuffer, training: bool) {
        for layer in &mut self.layers {
            layer.forward_propagate(&mut self.backend, inputs, training);
            inputs = layer.outputs()
        }
    }

    pub fn backward_propagate(&mut self, inputs: &WGPUBuffer, dataset: &WGPUBuffer) {
        let outputs = self.layers.last().unwrap().outputs();
        self.cost.prime(&mut self.backend, dataset, outputs);
        let mut d_outputs = &self.cost.d_inputs;

        for i in (1..self.layers.len()).rev() {
            let (left, right) = self.layers.split_at(i);
            let inputs = left.last().unwrap().outputs();
            right[0].backward_propagate(&mut self.backend, &inputs, d_outputs);
            d_outputs = right[0].d_inputs()
        }

        self.layers[0].backward_propagate(&mut self.backend, &inputs, d_outputs);
    }

    pub fn train(&mut self, datasets: Vec<Dataset>, epochs: usize, batches: usize, _rate: f32) {
        let mut epoch = 0;

        let mut gpu_datasets = Vec::new();
        for dataset in datasets {
            gpu_datasets.push(WGPUDataset {
                inputs: WGPUBuffer::from(&mut self.backend, dataset.inputs),
                outputs: WGPUBuffer::from(&mut self.backend, dataset.outputs),
            })
        }

        while epoch < epochs {
            let mut total = 0.0;
            for (i, dataset) in gpu_datasets.iter().enumerate() {
                self.forward_propagate(&dataset.inputs, true);
                self.backward_propagate(&dataset.inputs, &dataset.outputs);

                if !self.silent {
                    let outputs = self.layers.last().unwrap().outputs();
                    total += self
                        .cost
                        .cost(&mut self.backend, &outputs, &dataset.outputs);
                    let minibatch = outputs.shape[0];
                    if ((i + 1) * minibatch) % batches == 0 {
                        let cost = total / (batches) as f32;
                        let msg =
                            format!("Epoch={}, Dataset={}, Cost={}", epoch, i * minibatch, cost);
                        (self.logger.log)(msg);
                        total = 0.0;
                    }
                }
            }
            epoch += 1
        }
    }

    pub fn predict(&mut self, data: ArrayD<f32>) -> ArrayD<f32> {
        for layer in &mut self.layers {
            layer.reset(&mut self.backend, 1)
        }
        let inputs = WGPUBuffer::from(&mut self.backend, data);
        self.forward_propagate(&inputs, false);
        self.layers
            .last()
            .unwrap()
            .outputs()
            .read(&mut self.backend)
    }

    pub fn save(&mut self) -> Vec<u8> {
        let mut layers = Vec::new();
        for layer in &self.layers {
            layers.push(layer.save(&mut self.backend))
        }
        let mut tensors = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            match layer {
                Tensors::Dense(layer) => {
                    let weights = Tensor::new(layer.weights.view());
                    let biases = Tensor::new(layer.biases.view());
                    tensors.push((format!("{}w", i), weights));
                    tensors.push((format!("{}b", i), biases));
                }
                _ => {}
            }
        }
        let config = serde_json::to_string(&self.config).unwrap();
        let metadata = HashMap::from([("metadata".to_string(), config)]);
        serialize(tensors, &Some(metadata)).unwrap()
    }

    pub fn load(backend: WGPUBackend, buffer: &[u8], logger: Logger) -> Self {
        let tensors = SafeTensors::deserialize(buffer).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(buffer).unwrap();
        let data = metadata.metadata().as_ref().unwrap();
        let json = data.get("metadata").unwrap();
        let config: BackendConfig = serde_json::from_str(json).unwrap();
        let mut layers = Vec::new();

        for (i, layer) in config.layers.iter().enumerate() {
            match layer {
                Layer::Dense(_) => layers.push(Tensors::Dense(DenseTensors {
                    weights: to_arr(tensors.tensor(&format!("{}w", i)).unwrap()),
                    biases: to_arr(tensors.tensor(&format!("{}b", i)).unwrap()),
                })),
                _ => {}
            };
        }

        Backend::new(backend, config, logger, Some(layers))
    }
}
