import {
  DataType,
  ensureDataType,
  WebGPUBackend,
  WebGPUData,
} from "../../../deps.ts";
import { GPUMatrix } from "../matrix.ts";

let uniform: WebGPUData;

export async function feedForward<T extends DataType>(
  backend: WebGPUBackend,
  inputs: GPUMatrix<T>,
  weights: GPUMatrix<T>,
  biases: GPUMatrix<T>,
  outputs: GPUMatrix<T>,
  activation: string,
) {
  const type = ensureDataType(inputs.type, weights.type, outputs.type);
  const code = shader(type, activation);
  const pipeline = await backend.register(code);
  const buffer = new Uint32Array([inputs.x, outputs.x, inputs.y]);
  if (!uniform) {
    const usage = GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM;
    uniform = await WebGPUData.from(backend, buffer, "u32", usage);
  } else {
    backend.device.queue.writeBuffer(uniform.buffer, 0, buffer);
  }
  backend.execute(
    pipeline,
    [outputs.x, inputs.y, 1],
    [
      inputs.data,
      weights.data,
      biases.data,
      outputs.data,
      uniform,
    ],
  );
}

const shader = (type: DataType, activation: string) => `
struct Data {
  inputSize: u32,
  outputSize: u32,
  batches: u32
};

struct Matrix {
  values: array<${type}>
};

@group(0) @binding(0)
var<storage, read> inputs: Matrix;
@group(0) @binding(1)
var<storage, read> weights: Matrix;
@group(0) @binding(2)
var<storage, read> biases: Matrix;
@group(0) @binding(3)
var<storage, write> outputs: Matrix;

@group(0) @binding(4)
var<uniform> data: Data;

fn activation(weighted_sum: ${type}) -> ${type} {
  ${activation};
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= data.outputSize || global_id.y >= data.batches) {
    return;
  };

  var weighted_sum = ${type}(0);
  for (var k = 0u; k < data.inputSize; k += 1u) {
    var a = k + global_id.y * data.inputSize;
    var b = global_id.x + k * data.outputSize;    
    weighted_sum += inputs.values[a] * weights.values[b];
  };

  let idx = global_id.x + global_id.y * data.outputSize;
  outputs.values[idx] = activation(weighted_sum + biases.values[global_id.x]);
}
`;
