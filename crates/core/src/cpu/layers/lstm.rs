use crate::{sigmoid, CPUInit, CPURegularizer, Init, LSTMLayer, Tensors};
use core::f32;
use ndarray::{
    concatenate, s, Array2, Array3, ArrayD, Axis, Dimension, Ix2, Ix3, IxDyn,
};
use std::ops::AddAssign;
/// Indices
/// 0 - Input Gate
/// 1 - Forget Gate
/// 2 - Output Gate
/// 3 - Candidate Gate

pub struct LSTMCPULayer {
    pub output_size: IxDyn,
    pub inputs: Array3<f32>,
    pub return_sequences: bool,

    pub w_ih: Array3<f32>,
    pub w_hh: Array3<f32>,
    pub biases: Array2<f32>,

    pub d_w_ih: Array3<f32>,
    pub d_w_hh: Array3<f32>,
    pub d_biases: Array2<f32>,

    pub l_w_ih: Array3<f32>,
    pub l_w_hh: Array3<f32>,
    pub l_biases: Array2<f32>,

    pub h: Array3<f32>,
    pub c: Array3<f32>,

    pub regularizer: CPURegularizer,
}

impl LSTMCPULayer {
    pub fn new(config: LSTMLayer, size: IxDyn, _tensors: Option<Tensors>) -> Self {
        let return_sequences = config.return_sequences.unwrap_or(false);
        let init = CPUInit::from_default(config.init, Init::Uniform);
        let input_size = Ix3(size[0], size[1], size[2]);
        let weight_size = Ix3(4, size[2], config.size);
        let output_size = if return_sequences {
            IxDyn(&[size[0], size[1], config.size])
        } else {
            IxDyn(&[size[0], config.size])
        };

        Self {
            return_sequences,
            output_size,
            inputs: Array3::zeros(input_size),
            w_ih: init
                .init(weight_size.into_dyn(), size[2], config.size)
                .into_dimensionality::<Ix3>()
                .unwrap(),
            w_hh: init
                .init(IxDyn(&[4, config.size, config.size]), size[2], config.size)
                .into_dimensionality::<Ix3>()
                .unwrap(),
            biases: Array2::zeros((4, config.size)),
            d_w_ih: Array3::zeros(weight_size),
            d_w_hh: Array3::zeros((4, config.size, config.size)),
            d_biases: Array2::zeros((4, config.size)),
            l_w_ih: Array3::zeros(weight_size),
            l_w_hh: Array3::zeros((4, config.size, config.size)),
            l_biases: Array2::zeros((4, config.size)),
            h: Array3::zeros((size[1], size[0], config.size)),
            c: Array3::zeros((size[1], size[0], config.size)),
            regularizer: CPURegularizer::from(
                config.c.unwrap_or(0.0),
                config.l1_ratio.unwrap_or(1.0),
            ),
        }
    }

    pub fn output_size(&self) -> Vec<usize> {
        self.output_size.as_array_view().to_vec()
    }

    pub fn reset(&mut self, batches: usize) {
        self.inputs = Array3::zeros((batches, self.inputs.dim().1, self.inputs.dim().2));
        self.output_size[0] = batches;
    }

    pub fn forward_propagate(&mut self, inputs: ArrayD<f32>) -> ArrayD<f32> {
        self.inputs = inputs.into_dimensionality::<Ix3>().unwrap();
        let output_size = self.w_ih.shape()[2];
        self.h = Array3::zeros((self.inputs.shape()[1], self.inputs.shape()[0], output_size));
        self.c = Array3::zeros((self.inputs.shape()[1], self.inputs.shape()[0], output_size));
        let mut h_t = self.h.index_axis(Axis(0), 0).clone().to_owned();
        let mut c_t = self.c.index_axis(Axis(0), 0).clone().to_owned();

        let mut outputs = Array3::zeros(if self.return_sequences {
            (self.inputs.shape()[0], self.inputs.shape()[1], output_size)
        } else {
            (self.inputs.shape()[0], 1, output_size)
        });

        for t in 0..self.inputs.shape()[1] {
            let x_t = self
                .inputs
                .slice(s![.., t, ..])
                .to_owned()
                .into_dimensionality::<Ix2>()
                .unwrap(); // Current input
            let i_t = (&x_t.dot(&self.w_ih.index_axis(Axis(0), 0))
                + &h_t.dot(&self.w_hh.index_axis(Axis(0), 0))
                + &self.biases.index_axis(Axis(0), 0))
                .mapv(|x| sigmoid(&x));
            let f_t = (&x_t.dot(&self.w_ih.index_axis(Axis(0), 1))
                + &h_t.dot(&self.w_hh.index_axis(Axis(0), 1))
                + &self.biases.index_axis(Axis(0), 1))
                .mapv(|x| sigmoid(&x));
            let o_t = (&x_t.dot(&self.w_ih.index_axis(Axis(0), 2))
                + &h_t.dot(&self.w_hh.index_axis(Axis(0), 2))
                + &self.biases.index_axis(Axis(0), 2))
                .mapv(|x| sigmoid(&x));
            let g_t = (&x_t.dot(&self.w_ih.index_axis(Axis(0), 3))
                + &h_t.dot(&self.w_hh.index_axis(Axis(0), 3))
                + &self.biases.index_axis(Axis(0), 3))
                .mapv(|x| sigmoid(&x));

            c_t = &(&c_t * &f_t) + &(&g_t * &i_t);
            h_t = &c_t.mapv(|x| x.tanh()) * &o_t;

            self.h.index_axis_mut(Axis(0), t).assign(&h_t);
            self.c.index_axis_mut(Axis(0), t).assign(&c_t);

            if self.return_sequences {
                outputs.slice_mut(s![.., t, ..]).assign(&h_t);
            }
        }

        if self.return_sequences {
            outputs.into_dyn()
        }
        else {
            h_t.into_dyn()
        }
    }

    fn split_gates(
        &self,
        z: &Array2<f32>,
        hidden_size: usize,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let i_t = z
            .slice(ndarray::s![.., ..hidden_size])
            .mapv(|x| sigmoid(&x));
        let f_t = z
            .slice(ndarray::s![.., hidden_size..2 * hidden_size])
            .mapv(|x| sigmoid(&x));
        let o_t = z
            .slice(ndarray::s![.., 2 * hidden_size..3 * hidden_size])
            .mapv(|x| sigmoid(&x));
        let g_t = z
            .slice(ndarray::s![.., 3 * hidden_size..])
            .mapv(|x| sigmoid(&x));

        (i_t, f_t, o_t, g_t)
    }
    pub fn backward_propagate(&mut self, d_outputs: ArrayD<f32>) -> ArrayD<f32> {
        match d_outputs.shape().len() {
            2 => {
                let d_inputs = self.backward_propagate_2d(d_outputs.into_dimensionality::<Ix2>().unwrap());
                d_inputs.into_dyn()
            }
            3 => {
                let d_inputs = self.backward_propagate_3d(d_outputs.into_dimensionality::<Ix3>().unwrap());
                d_inputs.into_dyn()
            }
            _ => d_outputs
        }
    }
    pub fn backward_propagate_3d(&mut self, d_outputs: Array3<f32>) -> Array3<f32> {
        let sequence_length = self.inputs.shape()[1];
        let batch_size = d_outputs.shape()[0];
        let input_size = self.w_ih.shape()[1];
        let hidden_size = self.w_ih.shape()[2];

        self.d_w_ih = Array3::zeros((4, input_size, hidden_size));
        self.d_w_hh = Array3::zeros((4, hidden_size, hidden_size));
        self.d_biases = Array2::zeros((4, hidden_size));

        let h_prev = self.h.index_axis(Axis(0), sequence_length - 1);
        let c_prev = self.c.index_axis(Axis(0), sequence_length - 1);

        let mut d_inputs = Array3::<f32>::zeros((batch_size, sequence_length, input_size));

        let mut d_h_prev = Array2::<f32>::zeros((batch_size, hidden_size));
        let mut d_c_prev = Array2::<f32>::zeros((batch_size, hidden_size));

        let w_ih = concatenate(
            Axis(1),
            &[
                self.w_ih.index_axis(Axis(0), 0),
                self.w_ih.index_axis(Axis(0), 1),
                self.w_ih.index_axis(Axis(0), 2),
                self.w_ih.index_axis(Axis(0), 3),
            ],
        )
        .unwrap();
        let w_hh = concatenate(
            Axis(1),
            &[
                self.w_hh.index_axis(Axis(0), 0),
                self.w_hh.index_axis(Axis(0), 1),
                self.w_hh.index_axis(Axis(0), 2),
                self.w_hh.index_axis(Axis(0), 3),
            ],
        )
        .unwrap();
        for t in (0..sequence_length).rev() {
            let x_t = self
                .inputs
                .slice(s![.., t, ..])
                .into_dimensionality::<Ix2>()
                .unwrap();

            let d_h = d_outputs.slice(s![.., t, ..]).clone().to_owned().into_dimensionality::<Ix2>().unwrap();

            d_h_prev = d_h_prev + d_h;
            clip_gradients(&mut d_h_prev, 5f32);

            let tanned_c = c_prev.mapv(|x| x.tanh());

            let gates = x_t.dot(&w_ih)
                + h_prev.dot(&w_hh)
                + &self.biases.to_shape(4 * hidden_size).unwrap();
            let (i_t, f_t, o_t, g_t) = self.split_gates(&gates, hidden_size);

            let d_tanned_c = &d_h_prev * &o_t * (1.0 - &tanned_c.mapv(|x| x * x));
            let mut d_c_t = d_tanned_c + &d_c_prev;
            clip_gradients(&mut d_c_t, 5f32);

            let d_o_t = &d_h_prev * &c_prev.mapv(|x| x.tanh());
            let d_f_t = &d_c_t * &c_prev * &f_t * (1.0 - &f_t);
            let d_i_t = &d_c_t * &g_t * &i_t * (1.0 - &i_t);
            let d_g_t = &d_c_t * &i_t * (1.0 - g_t.mapv(|x| x * x));

            let d_gates = concatenate![Axis(1), d_i_t, d_f_t, d_o_t, d_g_t];
            d_inputs
                .slice_mut(s![.., t, ..])
                .assign(&d_gates.dot(&w_ih.t()));

            let d_gates_x = &d_gates.t().dot(&x_t);
            for (i, x) in d_gates_x
                .t()
                .axis_chunks_iter(Axis(1), hidden_size)
                .enumerate()
            {
                self.d_w_ih.index_axis_mut(Axis(0), i).add_assign(&x);
            }

            let d_gates_h = &d_gates.t().dot(&h_prev);

            for (i, x) in d_gates_h
                .t()
                .axis_chunks_iter(Axis(1), hidden_size)
                .enumerate()
            {
                self.d_w_hh.index_axis_mut(Axis(0), i).add_assign(&x);
            }
            self.d_biases += &d_gates
                .sum_axis(Axis(0))
                .to_shape((hidden_size, 4))
                .unwrap()
                .t();
            self.l_w_ih = self
                .regularizer
                .coeff(&(self.w_ih.clone().into_dyn()))
                .into_dimensionality::<Ix3>()
                .unwrap();
            self.l_w_hh = self
                .regularizer
                .coeff(&(self.w_hh.clone().into_dyn()))
                .into_dimensionality::<Ix3>()
                .unwrap();
            self.l_biases = self
                .regularizer
                .coeff(&(self.biases.clone().into_dyn()))
                .into_dimensionality::<Ix2>()
                .unwrap();
            d_h_prev = d_gates.dot(&w_hh.t());
            d_c_prev = d_c_t * f_t;
        }

        d_inputs
    }
    fn backward_propagate_2d(&mut self, d_outputs: Array2<f32>) -> Array3<f32> {
        let sequence_length = self.inputs.shape()[1];
        let batch_size = d_outputs.shape()[0];
        let input_size = self.w_ih.shape()[1];
        let hidden_size = self.w_ih.shape()[2];

        self.d_w_ih = Array3::zeros((4, input_size, hidden_size));
        self.d_w_hh = Array3::zeros((4, hidden_size, hidden_size));
        self.d_biases = Array2::zeros((4, hidden_size));

        let h_prev = self.h.index_axis(Axis(0), sequence_length - 1);
        let c_prev = self.c.index_axis(Axis(0), sequence_length - 1);

        let mut d_inputs = Array3::<f32>::zeros((batch_size, sequence_length, input_size));

        let mut d_h_prev = d_outputs.clone();
        let mut d_c_prev = Array2::<f32>::zeros((batch_size, hidden_size));

        let w_ih = concatenate(
            Axis(1),
            &[
                self.w_ih.index_axis(Axis(0), 0),
                self.w_ih.index_axis(Axis(0), 1),
                self.w_ih.index_axis(Axis(0), 2),
                self.w_ih.index_axis(Axis(0), 3),
            ],
        )
        .unwrap();
        let w_hh = concatenate(
            Axis(1),
            &[
                self.w_hh.index_axis(Axis(0), 0),
                self.w_hh.index_axis(Axis(0), 1),
                self.w_hh.index_axis(Axis(0), 2),
                self.w_hh.index_axis(Axis(0), 3),
            ],
        )
        .unwrap();

        for t in (0..sequence_length).rev() {
            let x_t = self
                .inputs
                .slice(s![.., t, ..])
                .into_dimensionality::<Ix2>()
                .unwrap();

            let tanned_c = c_prev.mapv(|x| x.tanh());

            let gates = x_t.dot(&w_ih)
                + h_prev.dot(&w_hh)
                + &self.biases.to_shape(4 * hidden_size).unwrap();
            let (i_t, f_t, o_t, g_t) = self.split_gates(&gates, hidden_size);

            let d_tanned_c = &d_h_prev * &o_t * (1.0 - &tanned_c.mapv(|x| x * x));
            let mut d_c_t = d_tanned_c + &d_c_prev;
            clip_gradients(&mut d_c_t, 5f32);

            let d_o_t = &d_h_prev * &c_prev.mapv(|x| x.tanh());
            let d_f_t = &d_c_t * &c_prev * &f_t * (1.0 - &f_t);
            let d_i_t = &d_c_t * &g_t * &i_t * (1.0 - &i_t);
            let d_g_t = &d_c_t * &i_t * (1.0 - g_t.mapv(|x| x * x));

            let d_gates = concatenate![Axis(1), d_i_t, d_f_t, d_o_t, d_g_t];
            //      println!("OT: {:?}\n\nFT {:?}", &d_h_prev, &d_c_prev);
            d_inputs
                .slice_mut(s![.., t, ..])
                .assign(&d_gates.dot(&w_ih.t()));

            let d_gates_x = &d_gates.t().dot(&x_t);
            for (i, x) in d_gates_x
                .t()
                .axis_chunks_iter(Axis(1), hidden_size)
                .enumerate()
            {
                self.d_w_ih.index_axis_mut(Axis(0), i).add_assign(&x);
            }

            let d_gates_h = &d_gates.t().dot(&h_prev);

            for (i, x) in d_gates_h
                .t()
                .axis_chunks_iter(Axis(1), hidden_size)
                .enumerate()
            {
                self.d_w_hh.index_axis_mut(Axis(0), i).add_assign(&x);
            }
            self.d_biases += &d_gates
                .sum_axis(Axis(0))
                .to_shape((hidden_size, 4))
                .unwrap()
                .t();
            self.l_w_ih = self
                .regularizer
                .coeff(&(self.w_ih.clone().into_dyn()))
                .into_dimensionality::<Ix3>()
                .unwrap();
            self.l_w_hh = self
                .regularizer
                .coeff(&(self.w_hh.clone().into_dyn()))
                .into_dimensionality::<Ix3>()
                .unwrap();
            self.l_biases = self
                .regularizer
                .coeff(&(self.biases.clone().into_dyn()))
                .into_dimensionality::<Ix2>()
                .unwrap();
            d_h_prev = d_gates.dot(&w_hh.t());
            d_c_prev = d_c_t * f_t;
            clip_gradients(&mut d_h_prev, 5f32);
        }

        d_inputs
    }
}

fn clip_gradients(grad: &mut Array2<f32>, threshold: f32) -> () {
    let norm = grad.mapv(|x| x.powi(2)).sum().sqrt();
    if norm > threshold {
        let scale = threshold / norm;
        grad.map_inplace(|x| *x *= scale);
    }
}
