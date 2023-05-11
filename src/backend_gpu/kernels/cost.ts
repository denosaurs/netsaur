import { GPUTensor, Rank, Shape, Shape3D } from "../../../core/types.ts";
import { WebGPUBackend } from "../../../deps.ts";

export async function cost(
  backend: WebGPUBackend,
  y: GPUTensor<Rank>,
  yHat: GPUTensor<Rank>,
  output: GPUTensor<Rank>,
  cost: string,
) {
  const code = shader_ff(cost, y.data.length);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [1, 1, 1],
    [y.data, yHat.data, output.data],
  );
}

const shader_ff = (cost: string, length: number) =>
  `struct Matrix {
  values: array<f32>
};

@group(0) @binding(0)
var<storage, read> y: Matrix;
@group(0) @binding(1)
var<storage, read> yHat: Matrix;
@group(0) @binding(2)
var<storage, write> output: Matrix;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var length = ${length}u;
    ${cost};
}`;

export async function prime(
  backend: WebGPUBackend,
  y: GPUTensor<Rank>,
  yHat: GPUTensor<Rank>,
  error: GPUTensor<Rank>,
  prime: string,
) {
  const shape = to3D(y.shape);
  const code = shader_bp(prime, shape);
  const pipeline = await backend.register(code);
  backend.execute(
    pipeline,
    [Math.ceil(shape[0] / 64), shape[1], shape[2]],
    [y.data, yHat.data, error.data],
  );
}

const shader_bp = (prime: string, shape: Shape3D) =>
  `struct Matrix {
values: array<f32>
};

@group(0) @binding(0)
var<storage, read> y: Matrix;
@group(0) @binding(1)
var<storage, read> yHat: Matrix;
@group(0) @binding(2)
var<storage, write> error: Matrix;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (
    global_id.x < ${shape[0]}u && 
    global_id.y < ${shape[1]}u && 
    global_id.z < ${shape[2]}u
  ) {
    var idx = global_id.x + global_id.y * ${shape[0]}u + global_id.z * ${
    shape[1]
  }u;
    ${prime};
  }
}`;

export const mse = `var sum = 0.0;
for (var i = 0u; i < length; i++) {
  sum += pow(y.values[i] - yHat.values[i], 2.0);
}
output.values[0] = sum / f32(length);`;

export const mse_prime =
  `error.values[idx] = y.values[idx] - yHat.values[idx];`;

function to3D(shape: Shape[Rank]): Shape3D {
  let res = shape as Shape3D;
  switch (shape.length) {
    case 2:
      res = [shape[0] * shape[1], 1, 1];
      break;
    case 4:
      res = [shape[0] * shape[1], shape[2], shape[3]];
  }
  return res;
}
