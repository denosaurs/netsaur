import {
  DataType,
  ensureType,
  WebGPUBackend,
  WebGPUData,
} from "../../../deps.ts";
import { GPUMatrix } from "../matrix.ts";

export async function feedForward<T extends DataType>(
  backend: WebGPUBackend,
  inputs: GPUMatrix<T>,
  weights: GPUMatrix<T>,
  products: GPUMatrix<T>,
  outputs: GPUMatrix<T>,
  activation: string,
) {
  const type = ensureType(inputs.type, weights.type, outputs.type);
  const code = shader(type, activation);
  const pipeline = await backend.register(code);
  const uniform = await WebGPUData.from(
    backend,
    new Uint32Array([inputs.x, outputs.x, inputs.y]),
    "u32",
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  );

  await backend.execute({
    pipeline,
    data: [inputs.data, weights.data, products.data, outputs.data, uniform],
    workgroups: [outputs.x, inputs.y, 1],
  });
}

const shader = (type: DataType, activation: string) => `
struct Data {
  inputSize: u32;
  outputSize: u32;
  batches: u32;
};

struct Matrix {
  values: array<${type}>;
};

[[group(0), binding(0)]]
var<storage, read> inputs: Matrix;
[[group(0), binding(1)]]
var<storage, read> weights: Matrix;
[[group(0), binding(2)]]
var<storage, write> products: Matrix;
[[group(0), binding(3)]]
var<storage, write> outputs: Matrix;

[[group(0), binding(4)]]
var<uniform> data: Data;

fn activation(weighted_sum: ${type}) -> ${type} {
  ${activation};
}

[[stage(compute), workgroup_size(8, 8, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
  if (global_id.x >= data.outputSize || global_id.y >= data.batches) {
    return;
  };

  var weighted_sum = ${type}(0);
  for (var k = 0u; k < data.inputSize; k = k + 1u) {
    var a = k + global_id.y * data.inputSize;
    var b = global_id.x + k * data.outputSize;    
    weighted_sum = weighted_sum + inputs.values[a] * weights.values[b];
  };

  let idx = global_id.x + global_id.y * data.outputSize;
  products.values[idx] = weighted_sum;
  outputs.values[idx] = activation(weighted_sum);
}
`;
