import { GPUInstance } from "../backends/gpu/mod.ts";
import { WebGPUData } from "../deps.ts";
import {
  BackendType,
  Rank,
  Shape,
  TensorData,
  TensorLike,
  TypedArray,
} from "./types.ts";
import { flatten, inferShape, iterate1D, toShape } from "./util.ts";

export class Tensor<R extends Rank, B extends BackendType> {
  static type: BackendType;
  shape: Shape[R];
  data: TensorData[B];
  rank = Rank.R1;

  get x() {
    return this.shape[0];
  }

  get y() {
    return this.shape[1] || 1;
  }

  get z() {
    return this.shape[2] || 1;
  }

  get w() {
    return this.shape[3] || 1;
  }

  constructor(data: TensorData[B], shape: Shape[R]) {
    this.shape = shape;
    this.data = data;
  }

  static zeroes<R extends Rank, B extends BackendType>(
    shape: Shape[R],
  ): Tensor<R, B> {
    const data = new Float32Array(toShape(shape, Rank.R1)[0]).fill(0);
    return new Tensor(data as TensorData[B], shape);
  }

  static ones<R extends Rank, B extends BackendType>(
    shape: Shape[R],
    value = 1,
  ): Tensor<R, B> {
    const data = new Float32Array(toShape(shape, Rank.R1)[0]).fill(value);
    return new Tensor(data as TensorData[B], shape);
  }

  to1D(): Tensor<Rank.R1, B> {
    return new Tensor(this.data, toShape(this.shape, Rank.R1));
  }

  to2D(): Tensor<Rank.R2, B> {
    return new Tensor(this.data, toShape(this.shape, Rank.R2));
  }

  to3D(): Tensor<Rank.R3, B> {
    return new Tensor(this.data, toShape(this.shape, Rank.R3));
  }

  get(...indices: number[]) {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * this.shape[i];
    }
    return (this.data as TensorData[BackendType.CPU])[index];
  }

  fmt() {
    const data = this.data as TensorData[BackendType.CPU];
    let res = "Tensor [\n";
    iterate1D(this.y, (i: number) => {
      res += "  [ ";
      const row = data.slice(i * this.x, (i + 1) * this.x);
      iterate1D(row.length, (j: number) => {
        res += row[j].toString() + ", ";
      });
      res += "]\n";
    });
    res += "]";
    return res;
  }
}

export function toData<B extends BackendType>(
  values: TensorLike,
): TensorData[B] {
  switch (Tensor.type) {
    case BackendType.CPU:
      return new Float32Array(flatten(values)) as TensorData[B];
    case BackendType.GPU: {
      const data = flatten(values);
      return new WebGPUData(
        GPUInstance.backend!,
        "f32",
        data.length,
      ) as TensorData[B];
    }
  }
}

export function tensor2D(
  values: TensorLike,
  shape?: Shape[Rank.R2],
) {
  const outputShape = shape || inferShape(values).slice();
  if (outputShape.length > 2) throw new Error("Invalid 2D Tensor");
  // values
  return new Tensor(toData(values), [outputShape[1], outputShape[0]]);
}

export function tensor1D(
  values: TensorLike,
  shape?: Shape[Rank.R1],
) {
  // deno-lint-ignore no-explicit-any
  if (Array.isArray((values as any)[0])) throw new Error("Invalid 1D Tensor");
  const outputShape = shape || [(values as TypedArray).length];
  return new Tensor(toData(values), outputShape);
}
