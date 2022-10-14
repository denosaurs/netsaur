import { TensorCPUBackend } from "../backends/cpu/tensor.ts";
import {  TensorLike, TensorBackend } from "./types.ts";
import {  inferShape } from "./util.ts";

export class Tensor {
  static backend: TensorBackend = new TensorCPUBackend();

  static setupBackend(backend: { tensor: TensorBackend}) {
    Tensor.backend = backend.tensor;
  }
}

export async function tensor2D(
  values: TensorLike,
) {
  const shape = inferShape(values).slice();
  if (shape.length > 2) throw new Error("Invalid 2D Tensor");
  // values
  return await Tensor.backend.tensor2D(values, shape[1], shape[0]);
}

export async function tensor1D(
  values: TensorLike
) {
  if (Array.isArray(values[0])) throw new Error("Invalid 1D Tensor");
  return await Tensor.backend.tensor1D(values);
}