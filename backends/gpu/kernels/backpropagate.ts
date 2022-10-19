import {
  DataType,
  ensureDataType,
  WebGPUBackend,
} from "../../../deps.ts";
import { GPUMatrix } from "../matrix.ts";

export async function backPropagate<T extends DataType>(
  backend: WebGPUBackend,
  inputs: GPUMatrix<T>,
  weights: GPUMatrix<T>,
  biases: GPUMatrix<T>,
  output: GPUMatrix<T>,
  cost: GPUMatrix<T>,
  error: GPUMatrix<T>,
  result: GPUMatrix<T>,
  prev: GPUMatrix<T>,
  rate: number,
  last: number,
  activation: string,
  costFn: string,
) {
  const type = ensureDataType(error.type, weights.type);
  const code = shader(type, activation, costFn, rate, weights.y, weights.x, prev.x, error.y, last);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [1, 1, 1],
    [
      inputs.data,
      weights.data,
      biases.data,
      output.data,
      cost.data,
      error.data,
      result.data,
      prev.data,
    ],
  );
}

const shader = (
  type: DataType,
  activation: string,
  cost: string,
  rate: number,
  input: number,
  output: number,
  prev: number,
  batches: number,
  layer: number,
) => `
  struct Matrix {
    values: array<${type}>,
  };
  
  @group(0) @binding(0)
  var<storage, read> inputs: Matrix;
  @group(0) @binding(1)
  var<storage, read_write> weights: Matrix;
  @group(0) @binding(2)
  var<storage, read_write> biases: Matrix;
  @group(0) @binding(3)
  var<storage, read> output: Matrix;
  @group(0) @binding(4)
  var<storage, read_write> cost: Matrix;
  @group(0) @binding(5)
  var<storage, read> error: Matrix;
  @group(0) @binding(6)
  var<storage, read_write> result: Matrix;
  @group(0) @binding(7)
  var<storage, read> prev: Matrix;
  
  fn activationPrime(output: ${type}) -> ${type} {
    ${activation};
  }

  fn costPrime(yHat: ${type}, y: ${type}) -> ${type} {
    ${cost};
  }
  
  @compute @workgroup_size(${output}, ${Math.max(batches, input)}, 1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.y < ${batches}u) {
      let idx = global_id.x + global_id.y * ${output}u;
      if (${layer == 0}) {
        result.values[idx] = costPrime(error.values[idx], output.values[idx]);
      } else {
        var weighted_sum = ${type}(0);
        for (var k = 0u; k < ${prev}u; k++) {
          var a = k + global_id.y * ${prev}u;
          var b = k + global_id.x * ${prev}u;    
          weighted_sum += prev.values[b] * error.values[a];
        };
        result.values[idx] = weighted_sum;
      }
      cost.values[idx] = result.values[idx] * activationPrime(output.values[idx]);
    };

    if (global_id.y < 1u) {
      for (var k = 0u; k < ${batches}u; k++) {
        let idx = global_id.x + k * ${output}u;
        biases.values[global_id.x] += cost.values[idx] * ${rate};
      }
    };

    if (global_id.y < ${input}u) {
      var weighted_sum = ${type}(0);
      for (var k = 0u; k < ${batches}u; k++) {
        var a = global_id.y + k * ${input}u;
        var b = global_id.x + k * ${output}u;    
        weighted_sum += cost.values[b] * inputs.values[a];
      };
      let idx = global_id.x + global_id.y * ${output}u;
      weights.values[idx] += weighted_sum * ${rate};
    };
  }
`;
