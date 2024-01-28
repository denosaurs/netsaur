use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use safetensors::{serialize, SafeTensors};

use crate::{
    to_arr, ActivationGPULayer, BackendConfig, BatchNorm1DGPULayer, BatchNorm2DGPULayer,
    BatchNormTensors, Conv2DGPULayer, ConvTensors, ConvTranspose2DGPULayer, Dataset, DenseGPULayer,
    DenseTensors, Dropout1DGPULayer, Dropout2DGPULayer, FlattenGPULayer, GPUCost, GPULayer,
    GPUOptimizer, GPUScheduler, GetTensor, Layer, Logger, Pool2DGPULayer, SoftmaxGPULayer, Tensor,
    Tensors,
};
pub use cudarc;

use crate::DType;

/// cudarc related errors
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error(transparent)]
    Compiler(#[from] cudarc::nvrtc::CompileError),

    #[error(transparent)]
    Cublas(#[from] cudarc::cublas::result::CublasError),

    #[error(transparent)]
    Curand(#[from] cudarc::curand::result::CurandError),

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{cuda} when loading {module_name}")]
    Load {
        cuda: cudarc::driver::DriverError,
        module_name: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    device: Arc<cudarc::driver::CudaDevice>,
    #[allow(dead_code)]
    blas: Arc<cudarc::cublas::CudaBlas>,
    #[allow(dead_code)]
    curand: Arc<Mutex<CudaRng>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl std::ops::Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaDevice>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl CudaDevice {
    pub fn new(ordinal: usize) -> Self {
        let device = cudarc::driver::CudaDevice::new(ordinal).unwrap();
        let blas = cudarc::cublas::CudaBlas::new(device.clone()).unwrap();
        let curand = cudarc::curand::CudaRng::new(299792458, device.clone()).unwrap();

        Self {
            id: DeviceId::new(),
            device,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
        }
    }

    pub fn cuda_device(&self) -> Arc<cudarc::driver::CudaDevice> {
        self.device.clone()
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }
}

pub struct Backend {
    pub silent: bool,
    pub config: BackendConfig,
    pub cuda_device: CudaDevice,
    pub layers: Vec<GPULayer>,
    pub size: Vec<usize>,
    pub cost: GPUCost,
    pub optimizer: GPUOptimizer,
    pub scheduler: GPUScheduler,
    pub logger: Logger,
}

impl Backend {
    pub fn new(config: BackendConfig, logger: Logger, mut tensors: Option<Vec<Tensors>>) -> Self {
        let mut layers = Vec::new();
        let mut size = config.size.clone();
        for layer in config.layers.iter() {
            match layer.clone() {
                Layer::Activation(config) => {
                    let layer = ActivationGPULayer::new(config, IxDyn(&size));
                    layers.push(GPULayer::Activation(layer));
                }
                Layer::Conv2D(config) => {
                    let layer = Conv2DGPULayer::new(config, IxDyn(&size), tensors.get());
                    size = layer.output_size().to_vec();
                    layers.push(GPULayer::Conv2D(layer));
                }
                Layer::ConvTranspose2D(config) => {
                    let layer = ConvTranspose2DGPULayer::new(config, IxDyn(&size), tensors.get());
                    size = layer.output_size().to_vec();
                    layers.push(GPULayer::ConvTranspose2D(layer));
                }
                Layer::BatchNorm1D(config) => {
                    let layer = BatchNorm1DGPULayer::new(config, IxDyn(&size), tensors.get());
                    layers.push(GPULayer::BatchNorm1D(layer));
                }
                Layer::BatchNorm2D(config) => {
                    let layer = BatchNorm2DGPULayer::new(config, IxDyn(&size), tensors.get());
                    layers.push(GPULayer::BatchNorm2D(layer));
                }
                Layer::Dropout1D(config) => {
                    let layer = Dropout1DGPULayer::new(config, IxDyn(&size));
                    layers.push(GPULayer::Dropout1D(layer));
                }
                Layer::Dropout2D(config) => {
                    let layer = Dropout2DGPULayer::new(config, IxDyn(&size));
                    layers.push(GPULayer::Dropout2D(layer));
                }
                Layer::Dense(config) => {
                    let layer = DenseGPULayer::new(config, IxDyn(&size), tensors.get());
                    size = layer.output_size().to_vec();
                    layers.push(GPULayer::Dense(layer));
                }
                Layer::Flatten(config) => {
                    let layer = FlattenGPULayer::new(config, IxDyn(&size));
                    size = layer.output_size().to_vec();
                    layers.push(GPULayer::Flatten(layer));
                }
                Layer::Pool2D(config) => {
                    let layer = Pool2DGPULayer::new(config, IxDyn(&size));
                    size = layer.output_size().to_vec();
                    layers.push(GPULayer::Pool2D(layer));
                }
                Layer::Softmax => {
                    let layer = SoftmaxGPULayer::new(IxDyn(&size));
                    layers.push(GPULayer::Softmax(layer));
                }
            }
        }
        let optimizer = GPUOptimizer::from(config.optimizer.clone(), &mut layers);
        let scheduler = GPUScheduler::from(&config.scheduler);
        let cost = GPUCost::from(config.cost.clone());
        let silent = config.silent.is_some_and(|x| x == true);
        let cuda_device = CudaDevice::new(0);
        Self {
            logger,
            cuda_device,
            silent,
            config,
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
                    let cost = total / (batches) as f32;
                    let msg = format!("Epoch={}, Dataset={}, Cost={}", epoch, i * minibatch, cost);
                    (self.logger.log)(msg);
                    total = 0.0;
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
                GPULayer::BatchNorm1D(layer) => {
                    let gamma = Tensor::new(layer.gamma.view().into_dyn());
                    let beta = Tensor::new(layer.beta.view().into_dyn());
                    let running_mean = Tensor::new(layer.running_mean.view().into_dyn());
                    let running_var = Tensor::new(layer.running_var.view().into_dyn());
                    tensors.push((format!("{}g", i), gamma));
                    tensors.push((format!("{}b", i), beta));
                    tensors.push((format!("{}m", i), running_mean));
                    tensors.push((format!("{}v", i), running_var));
                }
                GPULayer::BatchNorm2D(layer) => {
                    let gamma = Tensor::new(layer.gamma.view().into_dyn());
                    let beta = Tensor::new(layer.beta.view().into_dyn());
                    let running_mean = Tensor::new(layer.running_mean.view().into_dyn());
                    let running_var = Tensor::new(layer.running_var.view().into_dyn());
                    tensors.push((format!("{}g", i), gamma));
                    tensors.push((format!("{}b", i), beta));
                    tensors.push((format!("{}m", i), running_mean));
                    tensors.push((format!("{}v", i), running_var));
                }
                GPULayer::ConvTranspose2D(layer) => {
                    let weights = Tensor::new(layer.weights.view().into_dyn());
                    let biases = Tensor::new(layer.biases.view().into_dyn());
                    tensors.push((format!("{}w", i), weights));
                    tensors.push((format!("{}b", i), biases));
                }
                GPULayer::Conv2D(layer) => {
                    let weights = Tensor::new(layer.weights.view().into_dyn());
                    let biases = Tensor::new(layer.biases.view().into_dyn());
                    tensors.push((format!("{}w", i), weights));
                    tensors.push((format!("{}b", i), biases));
                }
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
