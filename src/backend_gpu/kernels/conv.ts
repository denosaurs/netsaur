import { GPUTensor, Rank, Shape2D, Shape4D } from "../../../core/types.ts";
import { WebGPUBackend } from "../../../deps.ts";

export async function feedforward(
  backend: WebGPUBackend,
  inputs: GPUTensor<Rank.R4>,
  weights: GPUTensor<Rank.R4>,
  biases: GPUTensor<Rank.R1>,
  outputs: GPUTensor<Rank.R4>,
  strides: Shape2D,
  padding: number,
) {
  const code = shader_ff(
    inputs.shape,
    weights.shape,
    outputs.shape,
    strides,
    padding,
  );
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [
      Math.ceil(outputs.x / 4),
      Math.ceil(outputs.y / 4),
      Math.ceil(outputs.z / 4),
    ],
    [inputs.data, weights.data, biases.data, outputs.data],
  );
}

const shader_ff = (
  inputs: Shape4D,
  weights: Shape4D,
  outputs: Shape4D,
  strides: Shape2D,
  padding: number,
) =>
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
var<storage, read_write> outputs: Matrix;

@compute @workgroup_size(4, 4, 4)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  if (
    global_id.x < ${outputs[0]}u && 
    global_id.y < ${outputs[1]}u &&
    global_id.z < ${outputs[2]}u
	) {
    for (var w = 0u; w < ${outputs[3]}u; w += 1u) {
			var sum = biases.values[0];
			for (var i = 0u; i < ${weights[0]}u; i += 1u) {
				for (var j = 0u; j < ${weights[1]}u; j += 1u) {
					for (var k = 0u; k < ${weights[2]}u; k += 1u) {
						var W = global_id.x * ${strides[0]}u + i;
						var H = global_id.y * ${strides[1]}u + j;
						var P = W + H * ${inputs[0]}u + k * ${inputs[1]}u + 
              w * ${inputs[2]}u;
						var K = i + j * ${weights[0]}u + k * ${weights[1]}u + 
							global_id.z * ${weights[2]}u;
						sum += inputs.values[P] * weights.values[K];
					}
				}
			}
      var idx = global_id.x + ${padding}u + (global_id.y + ${padding}u) 
      * ${outputs[0]}u + global_id.z * ${outputs[1]}u + w * ${outputs[2]}u;
      outputs.values[idx] = sum;
    };
  }
}`;
