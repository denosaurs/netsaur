use ndarray::ArrayD;

use crate::{Backend, BackendType, CPUBackend, Layer, Cost, Dataset};

pub struct NetworkConfig {
    pub size: &'static [usize],
    pub layers: Vec<Layer>,
    pub backend: BackendType,
    pub cost: Cost
}

pub struct Network {
    pub backend: Backend,
}

impl Network {
    pub fn new(config: NetworkConfig) -> Self {
        let backend = match config.backend {
            BackendType::CPU => Backend::CPU(CPUBackend::new(config)),
        };
        Self { backend }
    }

    pub fn train(&mut self, datasets: Vec<Dataset>, epochs: usize, rate: f32) {
        match &mut self.backend {
            Backend::CPU(backend) => backend.train(datasets, epochs, rate)
        }
    }

    pub fn predict(&mut self, data: ArrayD<f32>) -> ArrayD<f32> {
        match &mut self.backend {
            Backend::CPU(backend) => backend.predict(data)
        }
    }
}
