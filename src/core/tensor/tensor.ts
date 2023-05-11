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
import { inferShape, length } from "./util.ts";
import { TensorJSON } from "../../model/types.ts";

/**
 * A generic N-dimensional tensor.
 */
export class Tensor<R extends Rank> {
  shape: Shape[R];
  data: Float32Array;

  constructor(data: Float32Array, shape: Shape[R]) {
    this.shape = shape;
    this.data = data;
  }

  /**
   * Creates an empty tensor.
   */
  static zeroes<R extends Rank>(shape: Shape[R]): Tensor<R> {
    return new Tensor(new Float32Array(length(shape)), shape);
  }

  /**
   * Serialise a tensor into JSON.
   */
  toJSON() {
    const data = new Array(this.data.length).fill(1);
    this.data.forEach((value, i) => data[i] = value);
    return { data, shape: this.shape };
  }

  /**
   * Deserialise a tensor from JSON.
   */
  static fromJSON(tensor: TensorJSON): Tensor<Rank> {
    return new Tensor(new Float32Array(tensor.data), tensor.shape);
  }
}

/**
 * Create an nth rank tensor from the given nthD array and shape.
 */
export function tensor<R extends Rank>(values: Float32Array, shape: Shape[R]) {
  return new Tensor(values, shape);
}

/**
 * Create a 1D tensor from the given 1D array.
 */
export function tensor1D(values: Array1D) {
  const shape = inferShape(values) as Shape1D;
  return new Tensor(new Float32Array(values), shape);
}

/**
 * Create a 2D tensor from the given 2D array.
 */
export function tensor2D(values: Array2D) {
  const shape = inferShape(values) as Shape2D;
  return new Tensor(new Float32Array(values.flat(1)), shape);
}

/**
 * Create a 3D tensor from the given 3D array.
 */
export function tensor3D(values: Array3D) {
  const shape = inferShape(values) as Shape3D;
  return new Tensor(new Float32Array(values.flat(2)), shape);
}

/**
 * Create a 4D tensor from the given 4D array.
 */
export function tensor4D(values: Array4D) {
  const shape = inferShape(values) as Shape4D;
  return new Tensor(new Float32Array(values.flat(3)), shape);
}

/**
 * Create a 5D tensor from the given 5D array.
 */
export function tensor5D(values: Array5D) {
  const shape = inferShape(values) as Shape5D;
  return new Tensor(new Float32Array(values.flat(4)), shape);
}

/**
 * Create a 6D tensor from the given 6D array.
 */
export function tensor6D(values: Array6D) {
  const shape = inferShape(values) as Shape6D;
  return new Tensor(new Float32Array(values.flat(5)), shape);
}
