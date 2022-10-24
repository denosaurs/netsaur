import {
  Rank,
  ShapeMap,
  Size1D,
  Size2D,
  TensorBackend,
  TensorLike,
} from "./types.ts";
import { inferShape, sizeFromShape } from "./util.ts";
import { TensorCPUBackend } from "../backends/cpu/tensor.ts";

export class Tensor<R extends Rank> {
  static backend: TensorBackend = new TensorCPUBackend();
  shape: ShapeMap[R];
  // deno-lint-ignore no-explicit-any
  data: any; //Data;
  size: number;
  rank = Rank.R1;
  constructor(values: TensorLike, shape: ShapeMap[R]) {
    this.shape = shape;
    this.size = sizeFromShape(this.shape);
    this.data = values;
  }

  static dot() {
  }
}

export async function tensor2D(
  values: TensorLike,
  shape?: Size2D,
) {
  const outputShape = shape || inferShape(values).slice();
  if (outputShape.length > 2) throw new Error("Invalid 2D Tensor");
  // values
  return await Tensor.backend.tensor2D(values, outputShape[1], outputShape[0]);
}

export async function tensor1D(
  values: TensorLike,
) {
  // deno-lint-ignore no-explicit-any
  if (Array.isArray((values as any)[0])) throw new Error("Invalid 1D Tensor");
  return await Tensor.backend.tensor1D(values);
}

export async function zeros1D(
  shape: Size1D,
) {
  return await tensor1D(new Array(shape[0]).fill(0));
}
export async function zeros2D(
  shape: Size2D,
) {
  return await tensor1D(new Array(shape[0]).fill(new Array(shape[1]).fill(0)));
}
