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
}

/**
 * Create an nth rank tensor from the given nthD array and shape.
 * ```ts
 * tensor([1, 2, 3, 4], [2, 2]);
 * ```
 */
export function tensor<R extends Rank>(values: Float32Array, shape: Shape[R]) {
  return new Tensor(values, shape);
}

/**
 * Create a 1D tensor from the given 1D array.
 *
 * ```ts
 * tensor1D([1, 2, 3, 4]);
 * ```
 */
export function tensor1D(values: Array1D) {
  const shape = inferShape(values) as Shape1D;
  return new Tensor(new Float32Array(values), shape);
}

/**
 * Create a 2D tensor from the given 2D array.
 *
 * ```ts
 * tensor2D([
 *  [1, 2, 3, 4],
 *  [5, 6, 7, 8],
 * ]);
 * ```
 */
export function tensor2D(values: Array2D) {
  const shape = inferShape(values) as Shape2D;
  return new Tensor(new Float32Array(values.flat(1)), shape);
}

/**
 * Create a 3D tensor from the given 3D array.
 *
 * ```ts
 * tensor3D([
 *  [
 *    [1, 2, 3, 4],
 *    [5, 6, 7, 8],
 *  ],
 *  [
 *    [1, 2, 3, 4],
 *    [5, 6, 7, 8],
 *  ],
 * ]);
 * ```
 */
export function tensor3D(values: Array3D) {
  const shape = inferShape(values) as Shape3D;
  return new Tensor(new Float32Array(values.flat(2)), shape);
}

/**
 * Create a 4D tensor from the given 4D array.
 *
 * ```ts
 * tensor4D([
 *  [
 *    [
 *      [1, 2, 3],
 *      [4, 5, 6],
 *    ],
 *    [
 *      [1, 2, 3],
 *      [4, 5, 6],
 *    ],
 *  ],
 *  [
 *    [
 *      [1, 2, 3],
 *      [4, 5, 6],
 *    ],
 *    [
 *      [1, 2, 3],
 *      [4, 5, 6],
 *    ]
 *  ],
 * ]);
 * ```
 */
export function tensor4D(values: Array4D) {
  const shape = inferShape(values) as Shape4D;
  return new Tensor(new Float32Array(values.flat(3)), shape);
}

/**
 * Create a 5D tensor from the given 5D array.
 *
 * ```ts
 * tensor5D([
 *   [
 *     [
 *       [
 *         [1, 2, 3],
 *         [4, 5, 6],
 *       ],
 *       [
 *         [1, 2, 3],
 *         [4, 5, 6],
 *       ],
 *     ],
 *     [
 *       [
 *         [1, 2, 3],
 *         [4, 5, 6],
 *       ],
 *       [
 *         [1, 2, 3],
 *         [4, 5, 6],
 *       ],
 *     ],
 *   ],
 * ]);
 * ```
 */
export function tensor5D(values: Array5D) {
  const shape = inferShape(values) as Shape5D;
  return new Tensor(new Float32Array(values.flat(4)), shape);
}

/**
 * Create a 6D tensor from the given 6D array.
 * ```ts
 * tensor6D([
 *   [
 *     [
 *       [
 *         [
 *           [1, 2, 3],
 *           [4, 5, 6],
 *         ],
 *         [
 *           [1, 2, 3],
 *           [4, 5, 6],
 *         ],
 *       ],
 *       [
 *         [
 *           [1, 2, 3],
 *           [4, 5, 6],
 *         ],
 *         [
 *           [1, 2, 3],
 *           [4, 5, 6],
 *         ],
 *       ],
 *     ]
 *   ]
 * ]);
 * ```
 */
export function tensor6D(values: Array6D) {
  const shape = inferShape(values) as Shape6D;
  return new Tensor(new Float32Array(values.flat(5)), shape);
}
