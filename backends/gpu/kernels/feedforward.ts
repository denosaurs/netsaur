import {
  DataType,
  ensureDataType,
  WebGPUBackend,
} from "../../../deps.ts";
import { GPUMatrix } from "../matrix.ts";

export async function feedForward<T extends DataType>(
  backend: WebGPUBackend,
  inputs: GPUMatrix<T>,
  weights: GPUMatrix<T>,
  biases: GPUMatrix<T>,
  outputs: GPUMatrix<T>,
  activation: string,
) {
  const type = ensureDataType(inputs.type, weights.type, outputs.type);
  const code = shader(type, activation, inputs.x, outputs.x, inputs.y);
  const pipeline = await backend.register(code);
  // const buffer = new Uint32Array([inputs.x, outputs.x, inputs.y]);
  // if (!uniform) {
  //   const usage = GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM;
  //   uniform = await WebGPUData.from(backend, buffer, "u32", usage);
  // } else {
  //   backend.device.queue.writeBuffer(uniform.buffer, 0, buffer);
  // }
  backend.execute(
    pipeline,
    [1, 1, 1],
    [
      inputs.data,
      weights.data,
      biases.data,
      outputs.data,
      // uniform,
    ],
  );
}

const shader = (
  type: DataType,
  activation: string,
  input: number,
  output: number,
  batches: number,
) => `
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

fn activation(weighted_sum: ${type}) -> ${type} {
  ${activation};
}

@compute @workgroup_size(${output}, ${batches}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  var weighted_sum = ${type}(0);
  for (var k = 0u; k < ${input}u; k += 1u) {
    var a = k + global_id.y * ${input}u;
    var b = global_id.x + k * ${output}u;    
    weighted_sum += inputs.values[a] * weights.values[b];
  };

  let idx = global_id.x + global_id.y * ${output}u;
  outputs.values[idx] = activation(weighted_sum + biases.values[global_id.x]);
}
`;
