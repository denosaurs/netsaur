import { GPUInstance } from "../backends/gpu/mod.ts";
import { WebGPUData } from "../deps.ts";
import {
  BackendType,
  Rank,
  Shape,
  TensorData,
  TensorJSON,
  TensorLike,
  TypedArray,
} from "./types.ts";
import { flatten, inferShape, iterate1D, to1D, to2D, to3D } from "./util.ts";

export class Tensor<R extends Rank, B extends BackendType> {
  static type: BackendType;
  shape: Shape[R];
  data: TensorData[B];

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
    const data = new Float32Array(to1D(shape)[0]).fill(0);
    return new Tensor(toData(data), shape);
  }

  static ones<R extends Rank, B extends BackendType>(
    shape: Shape[R],
    value = 1,
  ): Tensor<R, B> {
    const data = new Float32Array(to1D(shape)[0]).fill(value);
    return new Tensor(toData(data), shape);
  }

  to1D(): Tensor<Rank.R1, B> {
    return new Tensor(this.data, to1D(this.shape));
  }

  to2D(): Tensor<Rank.R2, B> {
    return new Tensor(this.data, to2D(this.shape));
  }

  to3D(): Tensor<Rank.R3, B> {
    return new Tensor(this.data, to3D(this.shape));
  }

  get(...indices: number[]) {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * this.shape[i];
    }
    return (this.data as TensorData[BackendType.CPU])[index];
  }

  async getData() {
    switch (Tensor.type) {
      case BackendType.CPU:
      case BackendType.Native:
        return Array.from(this.data as TypedArray);
      case BackendType.GPU: {
        const data = await (this.data as WebGPUData).get();
        return Array.from(data);
      }
    }
  }

  setData(values: Float32Array) {
    switch (Tensor.type) {
      case BackendType.CPU:
        return (this.data as Float32Array).set(values);
      case BackendType.GPU: {
        const data = (this.data as TensorData[BackendType.GPU]).buffer;
        GPUInstance.backend!.device.queue.writeBuffer(data, 0, values);
      }
    }
  }

  async toJSON() {
    return {
      data: await this.getData(),
      shape: this.shape,
    };
  }

  static fromJSON<R extends Rank, B extends BackendType>(
    tensor: TensorJSON,
  ): Tensor<R, B> {
    return new Tensor(toData(tensor.data), tensor.shape as Shape[R]);
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
    case BackendType.Native:
    case BackendType.CPU:
      return new Float32Array(flatten(values)) as TensorData[B];
    case BackendType.GPU: {
      const data = new Float32Array(flatten(values));
      const res = new WebGPUData(GPUInstance.backend!, "f32", data.length);
      GPUInstance.backend!.device.queue.writeBuffer(res.buffer, 0, data);
      return res as TensorData[B];
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
