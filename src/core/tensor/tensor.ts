import {
  Array1D,
  Array2D,
  Array3D,
  Array4D,
  Array5D,
  Array6D,
  Rank,
  Shape,
  Shape1D,
  Shape2D,
  Shape3D,
  Shape4D,
  Shape5D,
  Shape6D,
} from "../api/shape.ts";
import { GPUInstance } from "../../backend_gpu/mod.ts";
import { WebGPUData } from "../../../deps.ts";
import { Engine } from "../engine.ts";
import { BackendType, TensorData, TensorJSON, TypedArray } from "../types.ts";
import { inferShape, length } from "./util.ts";

export type CPUTensor<R extends Rank> = Tensor<R, BackendType.CPU>;

export type GPUTensor<R extends Rank> = Tensor<R, BackendType.GPU>;

export class Tensor<R extends Rank, B extends BackendType> {
  shape: Shape[R];
  data: TensorData[B];

  constructor(data: TensorData[B], shape: Shape[R]) {
    this.shape = shape;
    this.data = data;
  }

  static zeroes<R extends Rank, B extends BackendType>(
    shape: Shape[R],
  ): Tensor<R, B> {
    switch (Engine.type) {
      case BackendType.CPU:
        return new Tensor(new Float32Array(length(shape)), shape);
      case BackendType.GPU: {
        const data = new WebGPUData(GPUInstance.backend!, "f32", length(shape));
        return new Tensor(data, shape);
      }
    }
  }

  static from<R extends Rank, B extends BackendType>(
    values: Float32Array,
    shape: Shape[R],
  ): Tensor<R, B> {
    switch (Engine.type) {
      case BackendType.CPU:
        return new Tensor(values, shape);
      case BackendType.GPU: {
        const data = new WebGPUData(GPUInstance.backend!, "f32", length(shape));
        GPUInstance.backend!.device.queue.writeBuffer(data.buffer, 0, values);
        return new Tensor(data, shape);
      }
    }
  }

  async getData() {
    switch (Engine.type) {
      case BackendType.CPU:
        return Array.from(this.data as TypedArray);
      case BackendType.GPU: {
        const data = await (this.data as WebGPUData).get();
        return Array.from(data as TypedArray);
      }
    }
  }

  setData(values: Float32Array) {
    switch (Engine.type) {
      case BackendType.CPU:
        return (this.data as Float32Array).set(values);
      case BackendType.GPU: {
        const data = (this.data as TensorData[BackendType.GPU]).buffer;
        GPUInstance.backend!.device.queue.writeBuffer(data, 0, values);
      }
    }
  }

  async toJSON() {
    return { data: await this.getData(), shape: this.shape };
  }

  static fromJSON(tensor: TensorJSON): Tensor<Rank, BackendType> {
    return new Tensor(new Float32Array(tensor.data), tensor.shape);
  }
}

export function tensor<R extends Rank>(values: Float32Array, shape: Shape[R]) {
  return Tensor.from(values, shape);
}

export function tensor1D(values: Array1D) {
  const shape = inferShape(values) as Shape1D;
  return Tensor.from(new Float32Array(values), shape);
}

export function tensor2D(values: Array2D) {
  const shape = inferShape(values) as Shape2D;
  return Tensor.from(new Float32Array(values.flat(1)), shape);
}

export function tensor3D(values: Array3D) {
  const shape = inferShape(values) as Shape3D;
  return Tensor.from(new Float32Array(values.flat(2)), shape);
}

export function tensor4D(values: Array4D) {
  const shape = inferShape(values) as Shape4D;
  return Tensor.from(new Float32Array(values.flat(3)), shape);
}

export function tensor5D(values: Array5D) {
  const shape = inferShape(values) as Shape5D;
  return Tensor.from(new Float32Array(values.flat(4)), shape);
}

export function tensor6D(values: Array6D) {
  const shape = inferShape(values) as Shape6D;
  return Tensor.from(new Float32Array(values.flat(5)), shape);
}
