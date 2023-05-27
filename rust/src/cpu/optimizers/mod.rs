mod sgd;

pub use sgd::*;

use crate::{CPULayer, Optimizer};

pub enum CPUOptimizer {
    SGD(CPUSGDOptimizer),
}

impl CPUOptimizer {
    pub fn from(optimizer: Optimizer) -> Self {
        match optimizer {
            Optimizer::SGD => CPUOptimizer::SGD(CPUSGDOptimizer::new())
        }
    }

    pub fn update_grads(&mut self, layer: &mut CPULayer, rate: f32) {
        let (params, grads) = match layer {
            CPULayer::Dense(dense) => (
                vec![
                    dense.weights.view_mut().into_dyn(),
                    dense.biases.view_mut().into_dyn(),
                ],
                vec![
                    dense.d_weights.view().into_dyn(),
                    dense.d_biases.view().into_dyn(),
                ],
            ),
            _ => return,
        };
        match self {
            CPUOptimizer::SGD(sgd) => sgd.update_grads(params, grads, rate),
        }
    }
}
