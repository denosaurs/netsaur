import { GPUTensor, Rank } from "../../../core/types.ts";
import { WebGPUBackend } from "../../../deps.ts";

export async function feedForward(
  backend: WebGPUBackend,
  inputs: GPUTensor<Rank.R2>,
  weights: GPUTensor<Rank.R2>,
  biases: GPUTensor<Rank.R2>,
  outputs: GPUTensor<Rank.R2>,
  activation: string,
) {
  const code = shader(activation, inputs.x, outputs.x, inputs.y);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [1, 1, 1],
    [
      inputs.data,
      weights.data,
      biases.data,
      outputs.data,
    ],
  );
}

const shader = (
  activation: string,
  input: number,
  output: number,
  batches: number,
) => `
struct Matrix {
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

fn activation(weighted_sum: f32) -> f32 {
  ${activation};
}

@compute @workgroup_size(${output}, ${batches}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  var weighted_sum = f32(0);
  for (var k = 0u; k < ${input}u; k += 1u) {
    var a = k + global_id.y * ${input}u;
    var b = global_id.x + k * ${output}u;    
    weighted_sum += inputs.values[a] * weights.values[b];
  };

  let idx = global_id.x + global_id.y * ${output}u;
  outputs.values[idx] = activation(weighted_sum + biases.values[global_id.x]);
}
`;
