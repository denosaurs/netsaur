import {
  ArrayMap,
  Rank,
  Shape,
  Shape2D,
  Shape3D,
  Shape4D,
} from "../api/shape.ts";
import { BackendType } from "../types.ts";
import { Tensor } from "./tensor.ts";

/**
 * Infer the shape of an array.
 */
export function inferShape(arr: ArrayMap): number[] {
  const shape = [];
  let elem: ArrayMap | number = arr;
  while (Array.isArray(elem)) {
    shape.push(elem.length);
    elem = elem[0];
  }
  return shape;
}

/**
 * return the length of a shape.
 */
export function length(shape: Shape[Rank]) {
  let length = 1;
  shape.forEach((i) => length *= i);
  return length;
}

/**
 * convert a shape to a given rank.
 */
export function toShape<R extends Rank>(shape: Shape[Rank], rank: R): Shape[R] {
  if (rank < shape.length) {
    const res = new Array(rank).fill(1);
    for (let i = 1; i < shape.length + 1; i++) {
      if (i < rank) {
        res[rank - i] = shape[shape.length - i];
      } else {
        res[0] *= shape[shape.length - i];
      }
    }
    return res as Shape[R];
  } else if (rank > shape.length) {
    const res = new Array(rank).fill(1);
    for (let i = 1; i < shape.length + 1; i++) {
      res[rank - i] = shape[shape.length - i];
    }
    return res as Shape[R];
  } else {
    return shape as Shape[R];
  }
}

/**
 * convert a shape to a 1D shape.
 */
export function to1D(shape: Shape[Rank]) {
  return toShape(shape, Rank.R1);
}

/**
 * convert a shape to a 2D shape.
 */
export function to2D(shape: Shape[Rank]) {
  return toShape(shape, Rank.R2);
}

/**
 * convert a shape to a 3D shape.
 */
export function to3D(shape: Shape[Rank]) {
  return toShape(shape, Rank.R3);
}

/**
 * convert a shape to a 4D shape.
 */
export function to4D(shape: Shape[Rank]) {
  return toShape(shape, Rank.R3);
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
export function iterate2D(
  mat: Tensor<Rank, BackendType> | Shape[Rank],
  callback: (i: number, j: number) => void,
): void {
  mat = (Array.isArray(mat) ? mat : mat.shape) as Shape2D;
  for (let i = 0; i < mat[0]; i++) {
    for (let j = 0; j < mat[1]; j++) {
      callback(i, j);
    }
  }
}

/**
 * iterate over a 3D array.
 */
export function iterate3D(
  mat: Tensor<Rank, BackendType> | Shape[Rank],
  callback: (i: number, j: number, k: number) => void,
): void {
  mat = (Array.isArray(mat) ? mat : mat.shape) as Shape3D;
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
export function iterate4D(
  mat: Tensor<Rank, BackendType> | Shape[Rank],
  callback: (i: number, j: number, k: number, l: number) => void,
): void {
  mat = (Array.isArray(mat) ? mat : mat.shape) as Shape4D;
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
