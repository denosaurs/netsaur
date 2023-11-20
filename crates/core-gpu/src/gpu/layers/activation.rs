use ndarray::{Dimension, IxDyn};

use crate::{ActivationLayer, GPUActivation, WGPUBackend, WGPUBuffer, WGPUKernel};

pub struct ActivationGPULayer {
    // data
    pub memoize_output: bool,
    pub outputs: WGPUBuffer,

    // gradients
    pub d_inputs: WGPUBuffer,

    // kernels
    pub forward_kernel: WGPUKernel,
    pub backward_kernel: WGPUKernel,
}

impl ActivationGPULayer {
    pub fn new(backend: &mut WGPUBackend, config: ActivationLayer, size: &mut IxDyn) -> Self {
        let activation = GPUActivation::from(config.activation);
        let forward_kernel = kernel_forward(backend, size.size(), activation.activate);
        let backward_kernel = kernel_backward(backend, size.size(), activation.prime);

        Self {
            memoize_output: GPUActivation::memoize_output(&activation.activation),
            outputs: WGPUBuffer::new(backend, size.clone()),
            d_inputs: WGPUBuffer::new(backend, size.clone()),
            forward_kernel,
            backward_kernel,
        }
    }

    pub fn reset(&mut self, backend: &mut WGPUBackend, batches: usize) {
        let output_size = self.outputs.shape.as_array_view()[1];
        self.outputs = WGPUBuffer::new(backend, IxDyn(&[batches, output_size]))
    }

    pub fn forward_propagate(&self, backend: &mut WGPUBackend, inputs: &WGPUBuffer) {
        backend.execute(&self.forward_kernel, vec![inputs, &self.outputs]);
    }

    pub fn backward_propagate(
        &self,
        backend: &mut WGPUBackend,
        inputs: &WGPUBuffer,
        d_outputs: &WGPUBuffer,
    ) {
        if self.memoize_output {
            backend.execute(
                &self.backward_kernel,
                vec![&self.outputs, d_outputs, &self.d_inputs],
            );
        } else {
            backend.execute(
                &self.backward_kernel,
                vec![inputs, d_outputs, &self.d_inputs],
            );
        };
    }
}

fn kernel_forward(backend: &mut WGPUBackend, size: usize, activation: String) -> WGPUKernel {
    let source = format!(
        "struct Matrix {{
            values: array<f32>
        }};
          
        @group(0) @binding(0)
        var<storage, read> inputs: Matrix;
        @group(0) @binding(1)
        var<storage, read_write> outputs: Matrix;
          
        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            if (global_id.x < {size}u) {{
                var x = inputs.values[global_id.x];
                outputs.values[global_id.x] = {activation};
            }}
        }}"
    );
    backend.register(source, ((size as f64 / 64.0).ceil() as u32, 1, 1))
}

fn kernel_backward(backend: &mut WGPUBackend, size: usize, activation: String) -> WGPUKernel {
    let source = format!(
        "struct Matrix {{
            values: array<f32>
        }};
          
        @group(0) @binding(0)
        var<storage, read> inputs: Matrix;
        @group(0) @binding(1)
        var<storage, read> d_outputs: Matrix;
        @group(0) @binding(2)
        var<storage, read_write> d_inputs: Matrix;
          
        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            if (global_id.x < {size}u) {{
                var d_output = d_outputs.values[global_id.x];
                var x = inputs.values[global_id.x];
                d_inputs.values[global_id.x] = {activation} * d_output;
            }}
        }}"
    );
    backend.register(source, ((size as f64 / 64.0).ceil() as u32, 1, 1))
}
