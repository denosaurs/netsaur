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
import { WebGPUData } from "../../../deps.ts";
import { TensorBackend } from "../engine.ts";
import { BackendType } from "../types.ts";
import { inferShape } from "./util.ts";
import { TensorJSON } from "../../model/types.ts";

/**
 * TensorData is the data type for the Tensor based on the backend.
 */
export interface TensorData {
  [BackendType.CPU]: Float32Array;
  [BackendType.GPU]: WebGPUData;
  [BackendType.WASM]: Float32Array;
}

/**
 * Tensor for the CPU backend.
 */
export type CPUTensor<R extends Rank> = Tensor<R, BackendType.CPU>;

/**
 * Tensor for the GPU backend.
 */
export type GPUTensor<R extends Rank> = Tensor<R, BackendType.GPU>;

/**
 * A generic N-dimensional tensor.
 */
export class Tensor<R extends Rank, B extends BackendType> {
  static backend: TensorBackend;
  shape: Shape[R];
  data: TensorData[B];

  constructor(data: TensorData[B], shape: Shape[R]) {
    this.shape = shape;
    this.data = data;
  }

  /**
   * Creates an empty tensor.
   */
  static zeroes<R extends Rank, B extends BackendType>(
    shape: Shape[R],
  ): Tensor<R, B> {
    return Tensor.backend.zeroes(shape);
  }

  /**
   * Creates a tensor from an array of values.
   */
  static from<R extends Rank, B extends BackendType>(
    values: Float32Array,
    shape: Shape[R],
  ): Tensor<R, B> {
    return Tensor.backend.from(values, shape);
  }

  /**
   * Get tensor data as an array of values.
   */
  async get() {
    return await Tensor.backend.get(this);
  }

  /**
   * Set tensor data from an array of values.
   */
  set(values: Float32Array) {
    Tensor.backend.set(this, values);
  }

  /**
   * Serialise a tensor into JSON.
   */
  async toJSON() {
    return { data: await this.get(), shape: this.shape };
  }

  /**
   * Deserialise a tensor from JSON.
   */
  static fromJSON(tensor: TensorJSON): Tensor<Rank, BackendType> {
    return Tensor.from(new Float32Array(tensor.data), tensor.shape);
  }
}

/**
 * Create an nth rank tensor from the given nthD array and shape. 
 */
export function tensor<R extends Rank>(values: Float32Array, shape: Shape[R]) {
  return Tensor.from(values, shape);
}

/**
 * Create a 1D tensor from the given 1D array.
 */
export function tensor1D(values: Array1D) {
  const shape = inferShape(values) as Shape1D;
  return Tensor.from(new Float32Array(values), shape);
}

/**
 * Create a 2D tensor from the given 2D array.
 */
export function tensor2D(values: Array2D) {
  const shape = inferShape(values) as Shape2D;
  return Tensor.from(new Float32Array(values.flat(1)), shape);
}

/**
 * Create a 3D tensor from the given 3D array.
 */
export function tensor3D(values: Array3D) {
  const shape = inferShape(values) as Shape3D;
  return Tensor.from(new Float32Array(values.flat(2)), shape);
}

/**
 * Create a 4D tensor from the given 4D array.
 */
export function tensor4D(values: Array4D) {
  const shape = inferShape(values) as Shape4D;
  return Tensor.from(new Float32Array(values.flat(3)), shape);
}

/**
 * Create a 5D tensor from the given 5D array.
 */
export function tensor5D(values: Array5D) {
  const shape = inferShape(values) as Shape5D;
  return Tensor.from(new Float32Array(values.flat(4)), shape);
}

/**
 * Create a 6D tensor from the given 6D array.
 */
export function tensor6D(values: Array6D) {
  const shape = inferShape(values) as Shape6D;
  return Tensor.from(new Float32Array(values.flat(5)), shape);
}
