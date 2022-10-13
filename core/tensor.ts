import { DataType, DataTypeArray } from "../deps.ts";
import { Rank, ShapeMap, Size2D, Size3D, TensorLike, TypedArray } from "./types.ts";
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

  flatten(): DataTypeArray {
    if (this.rank > 2) return this.data as DataTypeArray;
    return flatten(this.data as TypedArray) as DataTypeArray;
  }
}

export function tensor2D(
  values: TensorLike,
  shape?: ShapeMap[Rank.R2],
  dtype: DataType = "f32",
): { data: DataTypeArray; size: Size2D } {
  const _tensor = new Tensor<Rank.R2>(values, shape, dtype);
  return {
    data: _tensor.flatten(),
    size: {
      x: _tensor.shape[1],
      y: _tensor.shape[0],
    },
  };
}

export function tensor3D(
  values: TensorLike,
  shape?: ShapeMap[Rank.R3],
  dtype: DataType = "f32",
): { data: DataTypeArray; size: Size3D } {
  const _tensor = new Tensor<Rank.R3>(values, shape, dtype);
  return {
    data: _tensor.flatten(),
    size: {
      x: _tensor.shape[1],
      y: _tensor.shape[0],
      z: _tensor.shape[2]
    },
  };
}

