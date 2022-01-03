import { DataType, ensureType, WebGPUBackend, WebGPUData } from "../../deps.ts";

export async function matMul<T extends DataType>(
  backend: WebGPUBackend,
  inputs: WebGPUData<T>,
  weights: WebGPUData<T>,
  outputs: WebGPUData<T>,
  inputSize: number,
  outputSize: number,
  batches: number,
  activation: string
) {
  const type = ensureType(inputs.type, weights.type, outputs.type);
  const pipeline = await backend.register(shader(type, activation));
  const uniform = await WebGPUData.from(
    backend,
    new Uint32Array([inputSize, outputSize, batches]),
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  );

  await backend.execute({
    pipeline,
    data: [inputs, weights, outputs, uniform],
    workgroups: [outputSize, batches, 1],
  });
}

const shader = (type: DataType, activation: string) => `
[[block]]
struct Data {
  inputSize: u32;
  outputSize: u32;
  batches: u32;
};

[[block]]
struct Matrix {
  values: array<${type}>;
};

[[group(0), binding(0)]]
var<storage, read> inputs: Matrix;
[[group(0), binding(1)]]
var<storage, read> weights: Matrix;
[[group(0), binding(2)]]
var<storage, write> outputs: Matrix;

[[group(0), binding(3)]]
var<uniform> data: Data;

fn activation(weighted_sum: ${type}) -> ${type} {
  ${activation};
}

[[stage(compute), workgroup_size(8, 8, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
  if (global_id.x >= data.outputSize || global_id.y >= data.batches) {
    return;
  };

  var weighted_sum = 0${type};
  for (var k = 0u; k < data.inputSize; k = k + 1u) {
    var a = k + global_id.y * data.inputSize;
    var b = global_id.x + k * data.outputSize;    
    weighted_sum = weighted_sum + inputs.values[a] * weights.values[b];
  };

  let idx = global_id.x + global_id.y * data.outputSize;
  outputs.values[idx] = activation(weighted_sum);
}
`;
