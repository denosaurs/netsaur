import { GPUTensor, Rank } from "../../../core/types.ts";
import { WebGPUBackend } from "../../../deps.ts";

export async function feedforward(
  backend: WebGPUBackend,
  inputs: GPUTensor<Rank.R2>,
  weights: GPUTensor<Rank.R2>,
  biases: GPUTensor<Rank.R2>,
  outputs: GPUTensor<Rank.R2>,
) {
  const code = shader_ff(inputs.x, outputs.x, outputs.y);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [Math.ceil(outputs.x / 8), Math.ceil(outputs.y / 8), 1],
    [inputs.data, weights.data, biases.data, outputs.data],
  );
}

const shader_ff = (input: number, output: number, batches: number) =>
  `struct Matrix {
  values: array<f32>
};

@group(0) @binding(0)
var<storage, read> inputs: Matrix;
@group(0) @binding(1)
var<storage, read> weights: Matrix;
@group(0) @binding(2)
var<storage, read> biases: Matrix;
@group(0) @binding(3)
var<storage, write> outputs: Matrix;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  if (global_id.x < ${output}u && global_id.y < ${batches}u) {
    var weighted_sum = biases.values[global_id.x];
    for (var k = 0u; k < ${input}u; k += 1u) {
      var a = k + global_id.y * ${input}u;
      var b = global_id.x + k * ${output}u;    
      weighted_sum += inputs.values[a] * weights.values[b];
    };
  
    let idx = global_id.x + global_id.y * ${output}u;
    outputs.values[idx] = weighted_sum;
  }
}`;

export async function backpropagate(
  backend: WebGPUBackend,
  inputs: GPUTensor<Rank.R2>,
  weights: GPUTensor<Rank.R2>,
  biases: GPUTensor<Rank.R2>,
  dError: GPUTensor<Rank.R2>,
  dInputs: GPUTensor<Rank.R2>,
  rate: number,
) {
  const code = shader_bp(rate, inputs.x, dError.x, inputs.y);
  const pipeline = await backend.register(code);
  const x = Math.max(inputs.x, dError.x)
  const y = Math.max(dError.x, dError.y)
  backend.execute(
    pipeline,
    [Math.ceil(x / 8), Math.ceil(y / 8), 1],
    [
      inputs.data,
      weights.data,
      biases.data,
      dError.data,
      dInputs.data,
    ],
  );
}

const shader_bp = (
  rate: number,
  input: number,
  output: number,
  batches: number,
) => 
  `struct Matrix {
  values: array<f32>,
};

@group(0) @binding(0)
var<storage, read> inputs: Matrix;
@group(0) @binding(1)
var<storage, read_write> weights: Matrix;
@group(0) @binding(2)
var<storage, read_write> biases: Matrix;
@group(0) @binding(3)
var<storage, read_write> dError: Matrix;
@group(0) @binding(4)
var<storage, read_write> dInputs: Matrix;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x < ${input}u && global_id.y < ${batches}u) {
    var dInput = 0.0;
    for (var k = 0u; k < ${output}u; k++) {
      var a = k + global_id.x * ${output}u;
      var b = k + global_id.y * ${output}u;    
      dInput += dError.values[b] * weights.values[a];
    };
    let idx = global_id.x + global_id.y * ${input}u;
    dInputs.values[idx] = dInput;
  }

  if (global_id.x < ${output}u && global_id.y < 1u) {
    for (var k = 0u; k < ${batches}u; k++) {
      let idx = global_id.x + k * ${output}u;
      biases.values[global_id.x] -= dError.values[idx] * ${rate};
    }
  };

  if (global_id.x < ${input}u && global_id.y < ${output}u) {
    var dWeight = 0.0;
    for (var k = 0u; k < ${batches}u; k++) {
      var a = global_id.x + k * ${input}u;
      var b = global_id.y + k * ${output}u;    
      dWeight += dError.values[b] * inputs.values[a];
    };
    let idx = global_id.y + global_id.x * ${output}u;
    weights.values[idx] -= dWeight * ${rate};
  };
}
`;
