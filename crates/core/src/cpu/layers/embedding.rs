use ndarray::{Array2, ArrayD, Axis, Ix2, IxDyn};
use std::ops::AddAssign;

use crate::{CPUInit, CPURegularizer, EmbeddingLayer, Init};

pub struct EmbeddingCPULayer {
    pub input_size: IxDyn,
    pub input_indices: Vec<usize>,
    pub output_size: Vec<usize>,
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub embeddings: Array2<f32>,
    pub d_embeddings: Array2<f32>,
    // regularization
    pub l_embeddings: Array2<f32>,

    pub regularizer: CPURegularizer,
}

impl EmbeddingCPULayer {
    pub fn new(config: EmbeddingLayer, size: IxDyn) -> Self {
        let init = CPUInit::from(Init::Uniform);
        let output_size = vec![size[0], size[1], config.embedding_size];
        let embeddings = init
            .init(IxDyn(&[config.vocab_size, config.embedding_size]), 0, 0)
            .into_dimensionality::<Ix2>()
            .unwrap();
        let d_embeddings = Array2::zeros((config.vocab_size, config.embedding_size));
        Self {
            input_size: size,
            input_indices: vec![],
            output_size,
            vocab_size: config.vocab_size,
            embedding_size: config.embedding_size,
            embeddings,
            d_embeddings,
            l_embeddings: Array2::zeros((config.vocab_size, config.embedding_size)),
            regularizer: CPURegularizer::from(
                config.c.unwrap_or(0.0),
                config.l1_ratio.unwrap_or(1.0),
            ),
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.output_size.clone()
    }

    pub fn reset(&mut self, batches: usize) {
        self.output_size[0] = batches
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        let input_indices: Vec<usize> = inputs.iter().map(|&x| x as usize).collect();
        self.input_indices = input_indices.clone();
        let embeddings = self.embeddings.select(Axis(0), input_indices.as_slice());
        //        let output_size = IxDyn(&self.output_size);
        embeddings
            .into_shape_with_order(IxDyn(&[
                inputs.shape()[0],
                inputs.shape()[1],
                self.embedding_size,
            ]))
            .unwrap()
    }

    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        let indices = Array2::from_shape_vec(
            Ix2(d_outputs.shape()[0], self.input_size[1]),
            self.input_indices.clone(),
        )
        .unwrap();
        self.d_embeddings.fill(0.0);
        d_outputs
            .axis_iter(Axis(0))
            .zip(indices.axis_iter(Axis(0)))
            .for_each(|(rec, i)| {
                rec.axis_iter(Axis(0)).zip(i).for_each(|(grad, idx)| {
                    if idx != &0 {
                        self.d_embeddings
                            .index_axis_mut(Axis(0), *idx)
                            .add_assign(&grad);
                    }
                });
            });
        self.l_embeddings = self
            .regularizer
            .coeff(&self.embeddings.clone().into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap();
        let mut input_size = self.input_size.clone();
        input_size[0] = d_outputs.shape()[0];
        ArrayD::from_shape_vec(
            input_size,
            self.input_indices.iter().map(|x| *x as f32).collect(),
        )
        .unwrap()
    }
}
