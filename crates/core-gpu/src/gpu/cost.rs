use ndarray::{Dimension, IxDyn};

use crate::{Cost, WGPUBackend, WGPUBuffer, WGPUKernel};

pub struct GPUCost {
    pub d_inputs: WGPUBuffer,
    pub cost_kernel: WGPUKernel,
    pub prime_kernel: WGPUKernel,
}

impl GPUCost {
    pub fn from(backend: &mut WGPUBackend, cost: Cost, size: IxDyn) -> GPUCost {
        let (cost, prime) = match cost {
            Cost::MSE => (MSE, MSE_PRIME),
            _ => unimplemented!(),
        };
        GPUCost {
            d_inputs: WGPUBuffer::new(backend, size.clone()),
            cost_kernel: kernel_cost(backend, cost.to_string(), size.size()),
            prime_kernel: kernel_cost(backend, prime.to_string(), size.size()),
        }
    }

    pub fn cost(
        &self,
        backend: &mut WGPUBackend,
        dataset: &WGPUBuffer,
        outputs: &WGPUBuffer,
    ) -> f32 {
        backend.execute(&self.cost_kernel, vec![dataset, outputs, &self.d_inputs]);
        self.d_inputs.read(backend)[0]
    }

    pub fn prime(&self, backend: &mut WGPUBackend, dataset: &WGPUBuffer, outputs: &WGPUBuffer) {
        backend.execute(&self.prime_kernel, vec![dataset, outputs, &self.d_inputs]);
    }
}

const MSE: &str = "cost.values[global_id.x] = y.values[global_id.x] - y_hat.values[global_id.x];";

const MSE_PRIME: &str =
    "cost.values[global_id.x] = y.values[global_id.x] - y_hat.values[global_id.x];";

fn kernel_cost(backend: &mut WGPUBackend, cost: String, size: usize) -> WGPUKernel {
    let source = format!(
        "struct Matrix {{
            values: array<f32>
        }};
        
        @group(0) @binding(0)
        var<storage, read> y_hat: Matrix;
        @group(0) @binding(1)
        var<storage, read> y: Matrix;
        @group(0) @binding(2)
        var<storage, read_write> cost: Matrix;
        
        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            if (global_id.x < {size}u) {{
                {cost}
            }}
        }}"
    );
    backend.register(source, ((size as f64 / 64.0).ceil() as u32, 1, 1))
}
