use std::collections::HashMap;

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use safetensors::{serialize, SafeTensors};

use crate::{
    to_arr, ActivationCPULayer, BackendConfig, BatchNorm1DCPULayer, BatchNorm2DCPULayer,
    BatchNormTensors, CPUCost, CPULayer, CPUOptimizer, CPUScheduler, Conv2DCPULayer, ConvTensors,
    ConvTranspose2DCPULayer, Dataset, DenseCPULayer, DenseTensors, Dropout1DCPULayer,
    Dropout2DCPULayer, FlattenCPULayer, GetTensor, Layer, Logger, Pool2DCPULayer, SoftmaxCPULayer,
    Tensor, Tensors,
};

pub struct Backend {
    pub silent: bool,
    pub config: BackendConfig,
    pub tolerance: f32,
    pub patience: usize,
    pub layers: Vec<CPULayer>,
    pub size: Vec<usize>,
    pub cost: CPUCost,
    pub optimizer: CPUOptimizer,
    pub scheduler: CPUScheduler,
    pub logger: Logger,
}

impl Backend {
    pub fn new(config: BackendConfig, logger: Logger, mut tensors: Option<Vec<Tensors>>) -> Self {
        let mut layers = Vec::new();
        let mut size = config.size.clone();
        for layer in config.layers.iter() {
            match layer.clone() {
                Layer::Activation(config) => {
                    let layer = ActivationCPULayer::new(config, IxDyn(&size));
                    layers.push(CPULayer::Activation(layer));
                }
                Layer::Conv2D(config) => {
                    let layer = Conv2DCPULayer::new(config, IxDyn(&size), tensors.get());
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::Conv2D(layer));
                }
                Layer::ConvTranspose2D(config) => {
                    let layer = ConvTranspose2DCPULayer::new(config, IxDyn(&size), tensors.get());
                    size = layer.output_size().to_vec();
                    layers.push(CPULayer::ConvTranspose2D(layer));
                }
                Layer::BatchNorm1D(config) => {
                    let layer = BatchNorm1DCPULayer::new(config, IxDyn(&size), tensors.get());
                    layers.push(CPULayer::BatchNorm1D(layer));
                }
                Layer::BatchNorm2D(config) => {
                    let layer = BatchNorm2DCPULayer::new(config, IxDyn(&size), tensors.get());
                    layers.push(CPULayer::BatchNorm2D(layer));
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
                    let layer = DenseCPULayer::new(config, IxDyn(&size), tensors.get());
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
        let optimizer = CPUOptimizer::from(config.optimizer.clone(), &mut layers);
        let scheduler = CPUScheduler::from(&config.scheduler);
        let cost = CPUCost::from(config.cost.clone());
        let silent = config.silent.is_some_and(|x| x == true);
        let tolerance = config.tolerance.unwrap_or(0.0);
        let patience = config.patience.unwrap_or(0);
        Self {
            logger,
            silent,
            config,
            tolerance,
            patience,
            layers,
            cost,
            optimizer,
            scheduler,
            size,
        }
    }

    pub fn forward_propagate(
        &mut self,
        mut inputs: ArrayD<f32>,
        training: bool,
        layers: Option<Vec<usize>>,
    ) -> ArrayD<f32> {
        match layers {
            Some(layer_indices) => {
                for layer_index in layer_indices {
                    let layer = self.layers.get_mut(layer_index).expect(&format!("Layer #{} does not exist.", layer_index));
                    inputs = layer.forward_propagate(inputs, training);
                }
            }
            None => {
                for layer in &mut self.layers {
                    inputs = layer.forward_propagate(inputs, training);
                }
            }
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
        let mut best_cost = -1f32;
        let mut disappointments = 0;
        let mut best_net = self.save();
        let mut cost = 0f32;
        while epoch < epochs {
            let mut total = 0.0;
            for (i, dataset) in datasets.iter().enumerate() {
                let outputs = self.forward_propagate(dataset.inputs.clone(), true, None);
                self.backward_propagate(outputs.view(), dataset.outputs.view());
                self.optimizer
                    .update_grads(&mut self.layers, &self.scheduler, rate, epoch);
                total += (self.cost.cost)(outputs.view(), dataset.outputs.view());
                let minibatch = outputs.dim()[0];
                if !self.silent && ((i + 1) * minibatch) % batches == 0 {
                    cost = total / (batches) as f32;
                    let msg = format!("Epoch={}, Dataset={}, Cost={}", epoch, i * minibatch, cost);
                    (self.logger.log)(msg);
                    total = 0.0;
                }
            }
            if self.patience != 0 {
                if best_cost < 0.0 {
                    best_cost = cost;
                }
                if cost < best_cost - self.tolerance {
                    disappointments = 0;
                    best_cost = cost;
                    best_net = self.save();
                }  else {
                    disappointments += 1;
                    if !self.silent {
                        println!("Patience counter: {} disappointing epochs out of {}.", disappointments, self.patience);
                    }
                }
                if disappointments >= self.patience {
                    if !self.silent {
                        println!("No improvement for {} epochs. Stopping early at cost={}", disappointments, best_cost);
                    }
                    let net = Self::load(&best_net, Logger { log: |x| println!("{}", x) });
                    self.layers = net.layers;
                    break;
                }
            }
            epoch += 1
        }
    }

    pub fn predict(&mut self, data: ArrayD<f32>, layers: Option<Vec<usize>>) -> ArrayD<f32> {
        for layer in &mut self.layers {
            layer.reset(1);
        }
        self.forward_propagate(data, false, layers)
    }

    pub fn save(&self) -> Vec<u8> {
        let mut tensors = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                CPULayer::BatchNorm1D(layer) => {
                    let gamma = Tensor::new(layer.gamma.view().into_dyn());
                    let beta = Tensor::new(layer.beta.view().into_dyn());
                    let running_mean = Tensor::new(layer.running_mean.view().into_dyn());
                    let running_var = Tensor::new(layer.running_var.view().into_dyn());
                    tensors.push((format!("{}g", i), gamma));
                    tensors.push((format!("{}b", i), beta));
                    tensors.push((format!("{}m", i), running_mean));
                    tensors.push((format!("{}v", i), running_var));
                }
                CPULayer::BatchNorm2D(layer) => {
                    let gamma = Tensor::new(layer.gamma.view().into_dyn());
                    let beta = Tensor::new(layer.beta.view().into_dyn());
                    let running_mean = Tensor::new(layer.running_mean.view().into_dyn());
                    let running_var = Tensor::new(layer.running_var.view().into_dyn());
                    tensors.push((format!("{}g", i), gamma));
                    tensors.push((format!("{}b", i), beta));
                    tensors.push((format!("{}m", i), running_mean));
                    tensors.push((format!("{}v", i), running_var));
                }
                CPULayer::ConvTranspose2D(layer) => {
                    let weights = Tensor::new(layer.weights.view().into_dyn());
                    let biases = Tensor::new(layer.biases.view().into_dyn());
                    tensors.push((format!("{}w", i), weights));
                    tensors.push((format!("{}b", i), biases));
                }
                CPULayer::Conv2D(layer) => {
                    let weights = Tensor::new(layer.weights.view().into_dyn());
                    let biases = Tensor::new(layer.biases.view().into_dyn());
                    tensors.push((format!("{}w", i), weights));
                    tensors.push((format!("{}b", i), biases));
                }
                CPULayer::Dense(layer) => {
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

    pub fn load(buffer: &[u8], logger: Logger) -> Self {
        let tensors = SafeTensors::deserialize(buffer).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(buffer).unwrap();
        let data = metadata.metadata().as_ref().unwrap();
        let json = data.get("metadata").unwrap();
        let config: BackendConfig = serde_json::from_str(json).unwrap();
        let mut layers = Vec::new();

        for (i, layer) in config.layers.iter().enumerate() {
            match layer {
                Layer::BatchNorm1D(_) | Layer::BatchNorm2D(_) => {
                    layers.push(Tensors::BatchNorm(BatchNormTensors {
                        gamma: to_arr(tensors.tensor(&format!("{}g", i)).unwrap()),
                        beta: to_arr(tensors.tensor(&format!("{}b", i)).unwrap()),
                        running_mean: to_arr(tensors.tensor(&format!("{}m", i)).unwrap()),
                        running_var: to_arr(tensors.tensor(&format!("{}v", i)).unwrap()),
                    }))
                }
                Layer::Dense(_) => layers.push(Tensors::Dense(DenseTensors {
                    weights: to_arr(tensors.tensor(&format!("{}w", i)).unwrap()),
                    biases: to_arr(tensors.tensor(&format!("{}b", i)).unwrap()),
                })),
                Layer::Conv2D(_) | Layer::ConvTranspose2D(_) => {
                    layers.push(Tensors::Conv(ConvTensors {
                        weights: to_arr(tensors.tensor(&format!("{}w", i)).unwrap()),
                        biases: to_arr(tensors.tensor(&format!("{}b", i)).unwrap()),
                    }))
                }
                _ => {}
            };
        }

        Backend::new(config, logger, Some(layers))
    }
}
