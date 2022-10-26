import { DataType, DataTypeArray, DataTypeArrayConstructor } from "../deps.ts";
import {
  BackendType,
  Rank,
  Shape,
  Shape2D,
  TensorLike,
  TypedArray,
} from "./types.ts";
import { Tensor } from "./tensor.ts";

export const isTypedArray = (
  // deno-lint-ignore ban-types
  a: {},
): a is Float32Array | Int32Array | Uint8Array | Uint8ClampedArray =>
  a instanceof Float32Array || a instanceof Int32Array ||
  a instanceof Uint8Array || a instanceof Uint8ClampedArray;

export function getType(type: DataTypeArray<DataType>) {
  return (
    type instanceof Uint32Array
      ? "u32"
      : type instanceof Int32Array
        ? "i32"
        : "f32"
  );
}
export function fromType<T extends DataType>(type: string) {
  return (
    type === "u32"
      ? Uint32Array
      : type === "i32"
        ? Int32Array
        : type === "f32"
          ? Float32Array
          : Uint32Array
  ) as DataTypeArrayConstructor<T>;
}
export function toType<T extends DataType>(type: string) {
  return (
    type === "u32"
      ? Uint32Array
      : type === "i32"
        ? Int32Array
        : type === "f32"
          ? Float32Array
          : Uint32Array
  ) as DataTypeArrayConstructor<T>;
}

export class ActivationError extends Error {
  constructor(activation: string) {
    super(
      `Unknown activation function: ${activation}.  Available: "sigmoid", "tanh", "relu", "relu6" , "leakyrelu", "elu", "linear", "selu"`,
    );
  }
}

export const randomWeight = (): number => Math.random() * 0.4 - 0.2;

export const randomFloat = (min: number, max: number): number =>
  Math.random() * (max - min) + min;

export const gaussRandom = (): number => {
  if (gaussRandom.returnV) {
    gaussRandom.returnV = false;
    return gaussRandom.vVal;
  }
  const u = 2 * Math.random() - 1;
  const v = 2 * Math.random() - 1;
  const r = u * u + v * v;
  if (r === 0 || r > 1) {
    return gaussRandom();
  }
  const c = Math.sqrt((-2 * Math.log(r)) / r);
  gaussRandom.vVal = v * c;
  gaussRandom.returnV = true;
  return u * c;
};
gaussRandom.returnV = false;
gaussRandom.vVal = 0;

export const randomInteger = (min: number, max: number): number =>
  Math.floor(Math.random() * (max - min) + min);

export const randomN = (mu: number, std: number): number =>
  mu + gaussRandom() * std;
export const max = (
  values:
    | Float32Array
    | {
      [key: string]: number;
    },
): number =>
  (Array.isArray(values) || values instanceof Float32Array)
    ? Math.max(...values)
    : Math.max(...Object.values(values));

export const mse = (errors: Float32Array): number => {
  let sum = 0;
  for (let i = 0; i < errors.length; i++) {
    sum += errors[i] ** 2;
  }
  return sum / errors.length;
};

export function toShape<R extends Rank>(shape: Shape[Rank], rank: R): Shape[R] {
  if (rank < shape.length) {
    const res = new Array(rank).fill(1);
    res.forEach((_, i) => res[i] = shape[i]);
    res[rank - 1] = 1;
    for (let i = rank - 1; i < shape.length; i++) {
      res[rank - 1] *= shape[i];
    }
    return res as Shape[R];
  } else if (rank > shape.length) {
    const res = new Array(rank).fill(1);
    shape.map((val, i) => res[i] = val);
    return res as Shape[R];
  } else {
    return shape as Shape[R];
  }
}

export function to1D(shape: Shape[Rank]) {
  return toShape(shape, Rank.R1)
}

export function to2D(shape: Shape[Rank]) {
  return toShape(shape, Rank.R2)
}

export function to3D(shape: Shape[Rank]) {
  return toShape(shape, Rank.R3)
}

export function iterate2D(
  mat: Tensor<Rank.R2, BackendType> | Shape2D,
  callback: (i: number, j: number) => void,
): void {
  mat = (Array.isArray(mat) ? mat : mat.shape) as Shape2D;
  for (let i = 0; i < mat[0]; i++) {
    for (let j = 0; j < mat[1]; j++) {
      callback(i, j);
    }
  }
}

export function iterate1D(length: number, callback: (i: number) => void): void {
  for (let i = 0; i < length; i++) {
    callback(i);
  }
}

export function swap<T>(
  object: { [index: number]: T },
  left: number,
  right: number,
) {
  const temp = object[left];
  object[left] = object[right];
  object[right] = temp;
}

export function shuffle(
  // deno-lint-ignore no-explicit-any
  array: any[] | Uint32Array | Int32Array | Float32Array,
): void {
  let counter = array.length;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    swap(array, counter, index);
  }
}

export function shuffleCombo(
  // deno-lint-ignore no-explicit-any
  array: any[] | Uint32Array | Int32Array | Float32Array,
  // deno-lint-ignore no-explicit-any
  array2: any[] | Uint32Array | Int32Array | Float32Array,
): void {
  if (array.length !== array2.length) {
    throw new Error(
      `Array sizes must match to be shuffled together ` +
      `First array length was ${array.length}` +
      `Second array length was ${array2.length}`,
    );
  }
  let counter = array.length;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    swap(array, counter, index);
    swap(array2, counter, index);
  }
}
export function createShuffledIndices(n: number): Uint32Array {
  const shuffledIndices = new Uint32Array(n);
  for (let i = 0; i < n; ++i) {
    shuffledIndices[i] = i;
  }
  shuffle(shuffledIndices);
  return shuffledIndices;
}

export function randUniform(a: number, b: number) {
  const r = Math.random();
  return (b * r) + (1 - r) * a;
}

// export function flatten<
//   T extends number | Promise<number> | TypedArray,
// >(
//   arr: T,
//   result: T[] = [],
//   skipTypedArray = false,
// ): T[] | ArrayBufferLike {
//   if (result == null) {
//     result = [];
//   }
//   if (Array.isArray(arr) || isTypedArray(arr) && !skipTypedArray) {
//     for (let i = 0; i < arr.length; ++i) {
//       flatten(arr[i], result, skipTypedArray);
//     }
//   } else {
//     result.push(arr as T);
//   }
//   return result;
// }

export function flatten(input: TensorLike): number[] {
  // deno-lint-ignore no-explicit-any
  const stack = [...input as any];
  const res = [];
  while (stack.length) {
    const next = stack.pop()!;
    if (next.length) {
      stack.push(...next);
    } else {
      res.push(next);
    }
  }
  return res.reverse();
}

export function sizeFromShape(shape: number[]): number {
  if (shape.length === 0) {
    return 1;
  }
  let size = shape[0];
  for (let i = 1; i < shape.length; i++) {
    size *= shape[i];
  }
  return size;
}

export function computeStrides(shape: number[]): number[] {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

function createNestedArray(
  offset: number,
  shape: number[],
  a: TypedArray,
  isComplex = false,
) {
  // deno-lint-ignore no-array-constructor
  const ret = new Array();
  if (shape.length === 1) {
    const d = shape[0] * (isComplex ? 2 : 1);
    for (let i = 0; i < d; i++) {
      ret[i] = a[offset + i];
    }
  } else {
    const d = shape[0];
    const rest = shape.slice(1);
    const len = rest.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
    for (let i = 0; i < d; i++) {
      ret[i] = createNestedArray(offset + i * len, rest, a, isComplex);
    }
  }
  return ret;
}

export function toNestedArray(
  shape: number[],
  a: TypedArray,
  isComplex = false,
) {
  if (shape.length === 0) {
    return a[0];
  }
  const size = shape.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
  if (size === 0) {
    return [];
  }
  if (size !== a.length) {
    throw new Error(
      `[${shape}] does not match the input size ${a.length}${isComplex ? " for a complex tensor" : ""
      }.`,
    );
  }

  return createNestedArray(0, shape, a, isComplex);
}

export const average = (args: number[]) =>
  args.reduce((a, b) => a + b) / args.length;

export const maxIdx = (args: number[]) =>
  args.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

export function inferShape(val: TensorLike): number[] {
  let firstElem: typeof val = val;

  if (isTypedArray(val)) {
    return [val.length];
  }
  if (!Array.isArray(val)) {
    return [];
  }
  const shape: number[] = [];

  while (
    Array.isArray(firstElem) ||
    isTypedArray(firstElem)
  ) {
    shape.push(firstElem.length);
    firstElem = firstElem[0];
  }
  if (
    Array.isArray(val)
  ) {
    // TODO: assert shape
  }

  return shape;
}

export class Random {
  static #y2 = 0;
  static #previous_gaussian = false;
  static #randomStateProp = '_lcg_random_state';
  static #m = 4294967296;
  static #a = 1664525;
  static #c = 1013904223;

  static lcg(stateProperty: string) {
    // deno-lint-ignore no-explicit-any
    (Random as any)[stateProperty] = (Random.#a * (Random as any)[stateProperty] + Random.#c) % Random.#m;
    // deno-lint-ignore no-explicit-any
    return (Random as any)[stateProperty] / Random.#m;
  }

  static #lcgSeed(stateProperty: string, val = Math.random() * Random.#m) {
    // deno-lint-ignore no-explicit-any
    (Random as any)[stateProperty] = val >>> 0;
  }

  static setSeed(seed: number) {
    Random.#lcgSeed(Random.#randomStateProp, seed);
    Random.#previous_gaussian = false;
  }

  static random(min?: number | number[], max?: number | number[]): number {
    let rand;
    // deno-lint-ignore no-explicit-any
    if ((Random as any)[Random.#randomStateProp] != null) {
      rand = Random.lcg(Random.#randomStateProp);
    } else {
      rand = Math.random();
    }
    if (typeof min === 'undefined') {
      return rand;
    } else if (typeof max === 'undefined') {
      if (min instanceof Array) {
        return min[Math.floor(rand * min.length)];
      } else {
        return rand * min;
      }
    } else {
      if (min > max) {
        const tmp = min;
        min = max;
        max = tmp;
      }
      return rand * ((max as number) - (min as number)) + (min as number);
    }
  }

  static gaussian(mean: number, standard_deviation = 1) {
    let y1, x1, x2, w;
    if (Random.#previous_gaussian) {
      y1 = Random.#y2;
    } else {
      do {
        x1 = this.random(2) - 1;
        x2 = this.random(2) - 1;
        w = x1 * x1 + x2 * x2;
      } while (w >= 1);
      w = Math.sqrt(-2 * Math.log(w) / w);
      y1 = x1 * w;
      Random.#y2 = x2 * w;
      Random.#previous_gaussian = true;
    }

    const m = mean || 0;
    return y1 * standard_deviation + m;
  }
}