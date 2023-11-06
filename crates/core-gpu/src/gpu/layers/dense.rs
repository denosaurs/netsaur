use ndarray::{ArrayD, Dimension, IxDyn};

use crate::{
    DenseLayer, DenseTensors, GPUInit, Init, Tensors, WGPUBackend, WGPUBuffer, WGPUKernel,
};

pub struct DenseGPULayer {
    // data
    pub outputs: WGPUBuffer,

    // parameters
    pub weights: WGPUBuffer,
    pub biases: WGPUBuffer,

    // gradients
    pub d_weights: WGPUBuffer,
    pub d_biases: WGPUBuffer,
    pub d_inputs: WGPUBuffer,

    // kernels
    pub forward_kernel: WGPUKernel,
    pub backward_kernel: WGPUKernel,
}

impl DenseGPULayer {
    pub fn new(
        backend: &mut WGPUBackend,
        config: DenseLayer,
        size: &mut IxDyn,
        tensors: Option<Tensors>,
    ) -> Self {
        let init = GPUInit::from_default(config.init, Init::Uniform);
        let input_size = IxDyn(&[size[0], size[1]]);
        let weight_size = IxDyn(&[size[1], config.size[0]]);
        let bias_size = IxDyn(&[size[0]]);
        let output_size = IxDyn(&[size[0], config.size[0]]);
        *size = output_size.clone();

        let tensors = match tensors {
            Some(Tensors::Dense(tensors)) => tensors,
            _ => DenseTensors {
                weights: init.init(weight_size.clone(), input_size[1], config.size[0]),
                biases: ArrayD::zeros(config.size.clone()),
            },
        };

        let forward_kernel = kernel_forward(backend, input_size.clone(), output_size.clone());
        let backward_kernel = kernel_backward(backend, input_size.clone(), output_size.clone());

        Self {
            outputs: WGPUBuffer::new(backend, output_size),
            weights: WGPUBuffer::from(backend, tensors.weights),
            biases: WGPUBuffer::from(backend, tensors.biases),
            d_weights: WGPUBuffer::new(backend, weight_size),
            d_biases: WGPUBuffer::new(backend, bias_size),
            d_inputs: WGPUBuffer::new(backend, input_size),
            forward_kernel,
            backward_kernel,
        }
    }

    pub fn reset(&mut self, backend: &mut WGPUBackend, batches: usize) {
        let output_size = self.outputs.shape.as_array_view()[1];
        self.outputs = WGPUBuffer::new(backend, IxDyn(&[batches, output_size]))
    }

    pub fn forward_propagate(&self, backend: &mut WGPUBackend, inputs: &WGPUBuffer) {
        backend.execute(
            &self.forward_kernel,
            vec![inputs, &self.weights, &self.biases, &self.outputs],
        );
    }

    pub fn backward_propagate(
        &self,
        backend: &mut WGPUBackend,
        inputs: &WGPUBuffer,
        d_outputs: &WGPUBuffer,
    ) {
        backend.execute(
            &self.backward_kernel,
            vec![
                inputs,
                &self.weights,
                &self.biases,
                d_outputs,
                &self.d_inputs,
            ],
        );
    }

    pub fn save(&self, backend: &mut WGPUBackend) -> Tensors {
        Tensors::Dense(DenseTensors {
            weights: self.weights.read(backend),
            biases: self.weights.read(backend),
        })
    }
}

fn kernel_forward(backend: &mut WGPUBackend, input_size: IxDyn, output_size: IxDyn) -> WGPUKernel {
    let input = input_size[1];
    let output = output_size[1];
    let batches = input_size[0];
    let source = format!(
        "struct Matrix {{
            values: array<f32>
        }};
        
        @group(0) @binding(0)
        var<storage, read> inputs: Matrix;
        @group(0) @binding(1)
        var<storage, read> weights: Matrix;
        @group(0) @binding(2)
        var<storage, read> biases: Matrix;
        @group(0) @binding(3)
        var<storage, read_write> outputs: Matrix;
        
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            if (global_id.x < {output}u && global_id.y < {batches}u) {{
            var weighted_sum = biases.values[global_id.x];
            for (var k = 0u; k < {input}u; k += 1u) {{
                var a = k + global_id.y * {input}u;
                var b = global_id.x + k * {output}u;    
                weighted_sum += inputs.values[a] * weights.values[b];
            }};
            
            let idx = global_id.x + global_id.y * {output}u;
            outputs.values[idx] = weighted_sum;
            }}
        }}"
    );
    backend.register(
        source,
        (
            (output_size[1] as f64 / 8.0).ceil() as u32,
            (output_size[0] as f64 / 8.0).ceil() as u32,
            1,
        ),
    )
}

fn kernel_backward(backend: &mut WGPUBackend, input_size: IxDyn, output_size: IxDyn) -> WGPUKernel {
    let input = input_size[1];
    let output = output_size[1];
    let batches = input_size[0];
    let source = format!(
        "struct Matrix {{
            values: array<f32>,
        }};
        
        @group(0) @binding(0)
        var<storage, read> inputs: Matrix;
        @group(0) @binding(1)
        var<storage, read_write> weights: Matrix;
        @group(0) @binding(2)
        var<storage, read_write> biases: Matrix;
        @group(0) @binding(3)
        var<storage, read_write> d_outputs: Matrix;
        @group(0) @binding(4)
        var<storage, read_write> d_inputs: Matrix;
        
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            if (global_id.x < 3u && global_id.y < 4u) {{
                var d_input = 0.0;
                for (var k = 0u; k < 1u; k++) {{
                    var a = k + global_id.x * 1u;
                    var b = k + global_id.y * 1u;    
                    d_input = d_outputs.values[b] * weights.values[a];
                }};
                let idx = global_id.x + global_id.y * {input}u;
                d_inputs.values[idx] = d_input;
            }}
        
            if (global_id.x < {output}u && global_id.y < 1u) {{
                for (var k = 0u; k < {batches}u; k++) {{
                    let idx = global_id.x + k * {output}u;
                    biases.values[global_id.x] -= d_outputs.values[idx] * 0.1;
                }}
            }};
        
            if (global_id.x < {input}u && global_id.y < {output}u) {{
                var d_weight = 0.0;
                for (var k = 0u; k < {batches}u; k++) {{
                    var a = global_id.x + k * {input}u;
                    var b = global_id.y + k * {output}u;    
                    d_weight += d_outputs.values[b] * inputs.values[a];
                }};
                let idx = global_id.y + global_id.x * {output}u;
                weights.values[idx] -= d_weight * 0.1;
            }};
        }}"
    );
    let max_x = std::cmp::max(input_size[1], output_size[1]);
    let max_y = std::cmp::max(input_size[1], output_size[0]);
    backend.register(
        source,
        (
            (max_x as f64 / 8.0).ceil() as u32,
            (max_y as f64 / 8.0).ceil() as u32,
            1,
        ),
    )
}
