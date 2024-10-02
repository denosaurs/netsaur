import type { Tensor, Shape, NDArray, Order } from "./tensor.ts";
  

import type { DataType, DType, DTypeConstructor } from "./types.ts";

export function getDataType<DT extends DataType>(data: DType<DT>): DT {
  return (
    data instanceof Uint8Array
      ? "u8"
      : data instanceof Uint16Array
      ? "u16"
      : data instanceof Uint32Array
      ? "u32"
      : data instanceof Int8Array
      ? "i8"
      : data instanceof Int16Array
      ? "i16"
      : data instanceof Int32Array
      ? "i32"
      : data instanceof Float32Array
      ? "f32"
      : data instanceof Float64Array
      ? "f64"
      : "u8"
  ) as DT; // shouldn't reach "u8"
}

export function getConstructor<DT extends DataType>(
  dType: DT
): DTypeConstructor<DT> {
  switch (dType) {
    case "u8":
      return Uint8Array as DTypeConstructor<DT>;
    case "u16":
      return Uint16Array as DTypeConstructor<DT>;
    case "u32":
      return Uint32Array as DTypeConstructor<DT>;
    case "u64":
      return BigUint64Array as DTypeConstructor<DT>;
    case "i8":
      return Int8Array as DTypeConstructor<DT>;
    case "i16":
      return Int16Array as DTypeConstructor<DT>;
    case "i32":
      return Int32Array as DTypeConstructor<DT>;
    case "i64":
      return BigInt64Array as DTypeConstructor<DT>;
    case "f32":
      return Float32Array as DTypeConstructor<DT>;
    case "f64":
      return Float64Array as DTypeConstructor<DT>;
    default:
      throw new Error(`Unknown data type ${dType}.`);
  }
}

/**
 * Infer the shape of an array.
 */
export function inferShape<DT extends DataType>(arr: NDArray<DT>): number[] {
  const shape = [];
  let elem: NDArray<DT> | number | bigint = arr;
  while (Array.isArray(elem)) {
    shape.push(elem.length);
    elem = elem[0];
  }
  return shape;
}

/**
 * return the length of a shape.
 */
export function length(shape: Shape<Order>): number {
  let length = 1;
  shape.forEach((i) => (length *= i));
  return length;
}

/**
 * convert a shape to a given rank.
 */
export function toShape<O extends Order>(shape: Shape<Order>, rank: O): Shape<O> {
  if (rank < shape.length) {
    const res = new Array(rank).fill(1);
    for (let i = 1; i < shape.length + 1; i++) {
      if (i < rank) {
        res[rank - i] = shape[shape.length - i];
      } else {
        res[0] *= shape[shape.length - i];
      }
    }
    return res as Shape<O>;
  } else if (rank > shape.length) {
    const res = new Array(rank).fill(1);
    for (let i = 1; i < shape.length + 1; i++) {
      res[rank - i] = shape[shape.length - i];
    }
    return res as Shape<O>;
  } else {
    return shape as Shape<O>;
  }
}

/**
 * convert a shape to a 1D shape.
 */
export function to1D(shape: Shape<Order>): Shape<1> {
  return toShape(shape, 1);
}

/**
 * convert a shape to a 2D shape.
 */
export function to2D(shape: Shape<Order>): Shape<2> {
  return toShape(shape, 2);
}

/**
 * convert a shape to a 3D shape.
 */
export function to3D(shape: Shape<Order>): Shape<3> {
  return toShape(shape, 3);
}

/**
 * convert a shape to a 4D shape.
 */
export function to4D(shape: Shape<Order>): Shape<4> {
  return toShape(shape, 4);
}

/**
 * iterate over a 1D array.
 */
export function iterate1D(length: number, callback: (i: number) => void): void {
  for (let i = 0; i < length; i++) {
    callback(i);
  }
}

/**
 * iterate over a 2D array.
 */
export function iterate2D<DT extends DataType>(
  mat: Tensor<DT, 2> | Shape<2>,
  callback: (i: number, j: number) => void
): void {
  mat = (Array.isArray(mat) ? mat : mat.shape) as Shape<2>;
  for (let i = 0; i < mat[0]; i++) {
    for (let j = 0; j < mat[1]; j++) {
      callback(i, j);
    }
  }
}

/**
 * iterate over a 3D array.
 */
export function iterate3D<DT extends DataType>(
  mat: Tensor<DT, 3> | Shape<3>,
  callback: (i: number, j: number, k: number) => void
): void {
  mat = (Array.isArray(mat) ? mat : mat.shape) as Shape<3>;
  for (let i = 0; i < mat[0]; i++) {
    for (let j = 0; j < mat[1]; j++) {
      for (let k = 0; k < mat[2]; k++) {
        callback(i, j, k);
      }
    }
  }
}

/**
 * iterate over a 4D array.
 */
export function iterate4D<DT extends DataType>(
  mat: Tensor<DT, 4> | Shape<4>,
  callback: (i: number, j: number, k: number, l: number) => void
): void {
  mat = (Array.isArray(mat) ? mat : mat.shape) as Shape<4>;
  for (let i = 0; i < mat[0]; i++) {
    for (let j = 0; j < mat[1]; j++) {
      for (let k = 0; k < mat[2]; k++) {
        for (let l = 0; l < mat[3]; l++) {
          callback(i, j, k, l);
        }
      }
    }
  }
}
