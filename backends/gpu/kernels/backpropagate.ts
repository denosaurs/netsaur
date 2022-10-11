import {
  DataType,
  ensureDataType,
  WebGPUBackend,
  WebGPUData,
} from "../../../deps.ts";
import { GPUMatrix } from "../matrix.ts";

let uniform: WebGPUData;

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
  const code = shader(type, activation, costFn, rate);
  const pipeline = await backend.register(code);
  const buffer = new Uint32Array([weights.y, weights.x, prev.x, error.y, last]);
  if (!uniform) {
    const usage = GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM;
    uniform = await WebGPUData.from(backend, buffer, "u32", usage);
  } else {
    backend.device.queue.writeBuffer(uniform.buffer, 0, buffer);
  }
  backend.execute(
    pipeline,
    [weights.x, error.y, 1],
    [
      inputs.data,
      weights.data,
      biases.data,
      output.data,
      cost.data,
      error.data,
      result.data,
      prev.data,
      uniform,
    ],
  );
}

const shader = (
  type: DataType,
  activation: string,
  cost: string,
  rate: number,
) => `
  struct Data {
    inputSize: u32,
    outputSize: u32,
    prevSize: u32,
    batches: u32,
    layer: u32,
  };
  
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

  @group(0) @binding(8)
  var<uniform> data: Data;
  
  fn activationPrime(output: ${type}) -> ${type} {
    ${activation};
  }

  fn costPrime(yHat: ${type}, y: ${type}) -> ${type} {
    ${cost};
  }
  
  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x < data.outputSize && global_id.y < data.batches) {
      let idx = global_id.x + global_id.y * data.outputSize;
      if (data.layer == 0u) {
        result.values[idx] = costPrime(error.values[idx], output.values[idx]);
      } else {
        var weighted_sum = ${type}(0);
        for (var k = 0u; k < data.prevSize; k++) {
          var a = k + global_id.y * data.prevSize;
          var b = k + global_id.x * data.prevSize;    
          weighted_sum += prev.values[b] * error.values[a];
        };
        result.values[idx] = weighted_sum;
      }
      cost.values[idx] = result.values[idx] * activationPrime(output.values[idx]);
    };

    if (global_id.x < data.outputSize && global_id.y < 1u) {
      for (var k = 0u; k < data.batches; k++) {
        let idx = global_id.x + k * data.outputSize;
        biases.values[global_id.x] += cost.values[idx] * ${rate};
      }
    };

    if (global_id.x < data.outputSize && global_id.y < data.inputSize) {
      var weighted_sum = ${type}(0);
      for (var k = 0u; k < data.batches; k++) {
        var a = global_id.y + k * data.inputSize;
        var b = global_id.x + k * data.outputSize;    
        weighted_sum += cost.values[b] * inputs.values[a];
      };
      let idx = global_id.x + global_id.y * data.outputSize;
      weights.values[idx] += weighted_sum * ${rate};
    };
  }
`;
