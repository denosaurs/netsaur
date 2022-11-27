import { GPUTensor, Rank, Shape, Shape3D } from "../../../core/types.ts";
import { WebGPUBackend } from "../../../deps.ts";

export async function feedforward(
  backend: WebGPUBackend,
  inputs: GPUTensor<Rank>,
  outputs: GPUTensor<Rank>,
  activation: string,
) {
  const shape = to3D(inputs.shape)
  const code = shader_ff(activation, shape);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [Math.ceil(shape[0] / 64), shape[1], shape[2]], 
    [inputs.data, outputs.data]
  );
}

const shader_ff = (activation: string, shape: Shape3D) => 
`struct Matrix {
  values: array<f32>
};

@group(0) @binding(0)
var<storage, read> inputs: Matrix;
@group(0) @binding(1)
var<storage, write> outputs: Matrix;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (
    global_id.x < ${shape[0]}u && 
    global_id.y < ${shape[1]}u && 
    global_id.z < ${shape[2]}u
  ) {
    var idx = global_id.x + global_id.y * ${shape[0]}u + global_id.z * ${shape[1]}u;
    ${activation};
  }
}`;

export async function backpropagate(
  backend: WebGPUBackend,
  inputs: GPUTensor<Rank>,
  dError: GPUTensor<Rank>,
  dInputs: GPUTensor<Rank>,
  prime: string,
) {
  const shape = to3D(inputs.shape)
  const code = shader_bp(prime, shape);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [Math.ceil(shape[0] / 64), shape[1], shape[2]], 
    [inputs.data, dError.data, dInputs.data],
  );
}

const shader_bp = (prime: string, shape: Shape3D) => 
`struct Matrix {
  values: array<f32>
};

@group(0) @binding(0)
var<storage, read> inputs: Matrix;
@group(0) @binding(1)
var<storage, read> dError: Matrix;
@group(0) @binding(2)
var<storage, write> dInputs: Matrix;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (
    global_id.x < ${shape[0]}u && 
    global_id.y < ${shape[1]}u && 
    global_id.z < ${shape[2]}u
  ) {
    var idx = global_id.x + global_id.y * ${shape[0]}u + global_id.z * ${shape[1]}u;
    ${prime};
  }
}`;

export const sigmoid =
  `outputs.values[idx] = 1.0 / (1.0 + exp(-inputs.values[idx]));`;

export const sigmoid_prime =
  `var activation = inputs.values[idx] * (1.0 - inputs.values[idx]);
dInputs.values[idx] = activation * dError.values[idx];`;

function to3D(shape: Shape[Rank]): Shape3D {
  let res = shape as Shape3D
  switch (shape.length) {
    case 2:
      res = [shape[0] * shape[1], 1, 1];
      break
    case 4:
      res = [shape[0] * shape[1], shape[2], shape[3]];
  }
  return res;
}
