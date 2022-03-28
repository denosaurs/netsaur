import {
  DataType,
  ensureType,
  WebGPUBackend,
  WebGPUData,
} from "../../../deps.ts";
import { GPUMatrix } from "../matrix.ts";

export async function backPropagate<T extends DataType>(
  backend: WebGPUBackend,
  outputs: GPUMatrix<T>,
  result: GPUMatrix<T>,
  error: GPUMatrix<T>,
  cost: GPUMatrix<T>,
  weights: GPUMatrix<T>,
  inputs: GPUMatrix<T>,
  rate: number,
  activation: string,
  costFn: string,
) {
  const type = ensureType(outputs.type, result.type, error.type, cost.type);
  const code = shader(type, activation, costFn, rate);
  const pipeline = await backend.register(code);
  const uniform = await WebGPUData.from(
    backend,
    new Uint32Array([inputs.x, outputs.x, outputs.y, rate]),
    "u32",
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  );

  await backend.execute({
    pipeline,
    data: [
      outputs.data,
      result.data,
      error.data,
      cost.data,
      weights.data,
      inputs.data,
      uniform,
    ],
    workgroups: [outputs.x, outputs.y, 1],
  });
}

// embed rate into the shader to avoid adding another binding
// its not gonna change much anyways
const shader = (type: DataType, activation: string, cost: string, rate: number) => `
  struct Data {
    inputSize: u32;
    outputSize: u32;
    batches: u32;
  };
  
  struct Matrix {
    values: array<${type}>;
  };
  
  [[group(0), binding(0)]]
  var<storage, read> output: Matrix;
  [[group(0), binding(1)]]
  var<storage, read> result: Matrix;
  [[group(0), binding(2)]]
  var<storage, read_write> error: Matrix;
  [[group(0), binding(3)]]
  var<storage, read_write> cost: Matrix;
  [[group(0), binding(4)]]
  var<storage, read_write> weights: Matrix;
  [[group(0), binding(5)]]
  var<storage, read> inputs: Matrix;
  
  [[group(0), binding(6)]]
  var<uniform> data: Data;
  
  fn activationPrime(output: ${type}) -> ${type} {
    ${activation};
  }

  fn costPrime(yHat: ${type}, y: ${type}) -> ${type} {
    ${cost};
  }
  
  [[stage(compute), workgroup_size(8, 8, 1)]]
  fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let idx = global_id.x + global_id.y * data.outputSize;
    if (global_id.x < data.outputSize && global_id.y < data.batches) {
      var activation_prime = activationPrime(output.values[idx]);
      error.values[idx] = costPrime(result.values[idx], output.values[idx]);
      cost.values[idx] = activation_prime * error.values[idx];
    };

    if (global_id.x < data.inputSize && global_id.y < data.outputSize) {
      var weighted_sum = ${type}(0);
      for (var k = 0u; k < data.batches; k = k + 1u) {
        var a = global_id.y + k * data.inputSize;
        var b = global_id.x + k * data.outputSize;    
        weighted_sum = weighted_sum + inputs.values[a] * cost.values[b];
      };
      weights.values[idx] = weights.values[idx] + weighted_sum * ${rate};
    };
  }
`;
