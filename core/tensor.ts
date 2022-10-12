import { DataType } from "../deps.ts";
import { Rank, ShapeMap, TensorLike, TypedArray } from "./types.ts";
import { computeStrides, flatten, inferShape, sizeFromShape } from "./util.ts";

export class Tensor<R extends Rank = Rank> {
  readonly dtype: DataType;
  readonly rankType: R;
  readonly shape: ShapeMap[R];
  readonly size: number;
  readonly strides: number[];
  readonly data: TensorLike;
  constructor(
    values: TensorLike,
    shape?: ShapeMap[R],
    dtype: DataType = "f32",
  ) {
    this.shape = (shape ?? inferShape(values)).slice() as ShapeMap[R];
    this.data = values;
    this.dtype = dtype;
    this.size = sizeFromShape(this.shape);
    this.strides = computeStrides(this.shape);
    this.rankType = (this.rank < 5 ? this.rank.toString() : "higher") as R;
  }

  get rank(): number {
    return this.shape.length;
  }
  
  flatten() {
    if (this.rank > 2) return this.data;
    return flatten(this.data as TypedArray)
  }
}
