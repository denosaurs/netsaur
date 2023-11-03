mod adam;
mod sgd;

pub use adam::*;
use ndarray::{ArrayViewD, ArrayViewMutD};
pub use sgd::*;

use crate::{GPULayer, GPUScheduler, Optimizer};

pub enum GPUOptimizer {
    SGD(GPUSGDOptimizer),
    Adam(GPUAdamOptimizer),
}

impl GPUOptimizer {
    pub fn from(optimizer: Optimizer, layers: &mut Vec<GPULayer>) -> Self {
        let mut all_params = Vec::new();
        for layer in layers {
            if let Some((params, _)) = GPUOptimizer::get_params(layer) {
                all_params.push(params)
            }
        }
        match optimizer {
            Optimizer::SGD => GPUOptimizer::SGD(GPUSGDOptimizer::new()),
            Optimizer::Adam(config) => {
                GPUOptimizer::Adam(GPUAdamOptimizer::new(config, all_params))
            }
        }
    }

    pub fn update_grads(
        &mut self,
        layers: &mut Vec<GPULayer>,
        scheduler: &GPUScheduler,
        rate: f32,
        epoch: usize,
    ) {
        match self {
            GPUOptimizer::Adam(adam) => adam.t += 1.0,
            _ => {}
        }
        let mut idx = 0;
        for layer in layers.iter_mut() {
            if let Some((params, grads)) = GPUOptimizer::get_params(layer) {
                match self {
                    GPUOptimizer::SGD(sgd) => {
                        sgd.update_grads(params, grads, scheduler, rate, epoch)
                    }
                    GPUOptimizer::Adam(adam) => {
                        adam.update_grads(params, grads, idx, scheduler, rate)
                    }
                }
                idx += 1;
            }
        }
    }

    pub fn get_params<'a>(
        layer: &'a mut GPULayer,
    ) -> Option<(Vec<ArrayViewMutD<'a, f32>>, Vec<ArrayViewD<'a, f32>>)> {
        match layer {
            GPULayer::Dense(layer) => Some((
                vec![
                    layer.weights.view_mut().into_dyn(),
                    layer.biases.view_mut().into_dyn(),
                ],
                vec![
                    layer.d_weights.view().into_dyn(),
                    layer.d_biases.view().into_dyn(),
                ],
            )),
            _ => return None,
        }
    }
}
