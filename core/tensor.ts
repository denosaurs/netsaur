import { Rank, ShapeMap, Size2D, TensorLike } from "./types.ts";
import { inferShape } from "./util.ts";

export class Tensor<R extends Rank, > {
  shape: ShapeMap[R];
  data: Data;
  size: number;
  rank = Rank.R1;
  constructor(values: TensorLike, shape?: ShapeMap[R]) {
    this.shape = shape || inferShape(values);
    this.size = sizeFromShape(this.shape);
  }

  static dot() {
    
  }
}

export async function tensor2D(
  values: TensorLike,
  shape?: Size2D | [number, number],
) {
  const outputShape = shape === undefined
    ? inferShape(values).slice()
    : shape instanceof Array
    ? shape
    : [shape.y, shape.x];
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
