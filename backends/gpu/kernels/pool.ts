import { GPUTensor, Rank, Shape2D, Shape4D } from "../../../core/types.ts";
import { WebGPUBackend } from "../../../deps.ts";

export async function feedforward_max(
  backend: WebGPUBackend,
  inputs: GPUTensor<Rank.R4>,
  indices: GPUTensor<Rank.R4>,
  outputs: GPUTensor<Rank.R4>,
  strides: Shape2D,
) {
  const code = shader_ff_max(inputs.shape, outputs.shape, strides);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [
      Math.ceil(outputs.x / 4),
      Math.ceil(outputs.y / 4),
      Math.ceil(outputs.z / 4),
    ],
    [inputs.data, indices.data, outputs.data],
  );
}

// var len = i + j * ${strides[0] * strides[1]}u;

const shader_ff_max = (
  inputs: Shape4D,
  outputs: Shape4D,
  strides: Shape2D,
) =>
  `struct Matrix {
  values: array<f32>
};

@group(0) @binding(0)
var<storage, read> inputs: Matrix;
@group(0) @binding(1)
var<storage, write> indices: Matrix;
@group(0) @binding(2)
var<storage, write> outputs: Matrix;

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
      var len = 0u;
			var pool: array<f32, ${strides[0] * strides[1]}>;
			var index: array<u32, ${strides[0] * strides[1]}>;
			for (var i = 0u; i < ${strides[0]}u; i += 1u) {
				for (var j = 0u; j < ${strides[1]}u; j += 1u) {
					var W = global_id.x * ${strides[0]}u + i;
					var H = global_id.y * ${strides[1]}u + j;
					var idx = W + H * ${inputs[0]}u + global_id.z * 
						${inputs[1]}u + w * ${inputs[2]}u;
					pool[len] = inputs.values[idx];
					index[len] = idx;
          len += 1u;
				};
			};
      var idx = global_id.x + global_id.y * ${outputs[0]}u + 
				global_id.z * ${outputs[1]}u + w * ${outputs[2]}u;
			var max = 0u;
			for (var i = 0u; i < ${strides[0] * strides[1]}u; i += 1u) {
				if (pool[i] > pool[max]) {
					max = i;
				}
			};
			indices.values[idx] = f32(index[max]);
			outputs.values[idx] = pool[max];
    };
  }
}`;
