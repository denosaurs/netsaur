import { TensorCPUBackend } from "../backends/cpu/tensor.ts";
import { TensorBackend, TensorLike } from "./types.ts";
import { inferShape } from "./util.ts";

export class Tensor {
  static backend: TensorBackend = new TensorCPUBackend();
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
  values: TensorLike,
) {
  // deno-lint-ignore no-explicit-any
  if (Array.isArray((values as any)[0])) throw new Error("Invalid 1D Tensor");
  return await Tensor.backend.tensor1D(values);
}
