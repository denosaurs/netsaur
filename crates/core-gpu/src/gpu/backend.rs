use std::collections::HashMap;

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use safetensors::{serialize, SafeTensors};

use crate::{
    to_arr, ActivationGPULayer, BackendConfig, Dataset, DenseGPULayer, DenseTensors, GPUCost,
    GPULayer, GPUOptimizer, GPUScheduler, GetTensor, Layer, Logger, Tensor, Tensors, WGPUBackend,
};

pub struct Backend {
    pub backend: WGPUBackend,
    pub silent: bool,
    pub config: BackendConfig,
    pub layers: Vec<GPULayer>,
    pub size: Vec<usize>,
    pub cost: GPUCost,
    pub optimizer: GPUOptimizer,
    pub scheduler: GPUScheduler,
    pub logger: Logger,
}

impl Backend {
    pub fn new(
        backend: WGPUBackend,
        config: BackendConfig,
        logger: Logger,
        mut tensors: Option<Vec<Tensors>>,
    ) -> Self {
        let mut layers = Vec::new();
        let mut size = config.size.clone();
        for layer in config.layers.iter() {
            match layer.clone() {
                Layer::Activation(config) => {
                    let layer = ActivationGPULayer::new(config, IxDyn(&size));
                    layers.push(GPULayer::Activation(layer));
                }
                Layer::Dense(config) => {
                    let layer = DenseGPULayer::new(config, IxDyn(&size), tensors.get());
                    size = layer.output_size().to_vec();
                    layers.push(GPULayer::Dense(layer));
                }
                _ => unimplemented!(),
            }
        }
        let optimizer = GPUOptimizer::from(config.optimizer.clone(), &mut layers);
        let scheduler = GPUScheduler::from(&config.scheduler);
        let cost = GPUCost::from(config.cost.clone());
        let silent = config.silent.is_some_and(|x| x == true);
        Self {
            backend,
            logger,
            silent,
            config,
            layers,
            cost,
            optimizer,
            scheduler,
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
    ) -> ArrayD<f32> {
        let mut d_outputs = (self.cost.prime)(data, outputs);
        for layer in self.layers.iter_mut().rev() {
            d_outputs = layer.backward_propagate(d_outputs);
        }
        d_outputs
    }

    pub fn train(&mut self, datasets: Vec<Dataset>, epochs: usize, batches: usize, rate: f32) {
        let mut epoch = 0;
        while epoch < epochs {
            let mut total = 0.0;
            for (i, dataset) in datasets.iter().enumerate() {
                let outputs = self.forward_propagate(dataset.inputs.clone(), true);
                self.backward_propagate(outputs.view(), dataset.outputs.view());
                self.optimizer
                    .update_grads(&mut self.layers, &self.scheduler, rate, epoch);
                total += (self.cost.cost)(outputs.view(), dataset.outputs.view());
                let minibatch = outputs.dim()[0];
                if !self.silent && ((i + 1) * minibatch) % batches == 0 {
                    let cost = total / (batches) as f32;
                    let msg = format!("Epoch={}, Dataset={}, Cost={}", epoch, i * minibatch, cost);
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
                GPULayer::Dense(layer) => {
                    let weights = Tensor::new(layer.weights.view().into_dyn());
                    let biases = Tensor::new(layer.biases.view().into_dyn());
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
