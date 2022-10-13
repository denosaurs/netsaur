import { DataType, DataTypeArray, DataTypeArrayConstructor } from "../deps.ts";
import { CPUMatrix } from "../backends/cpu/matrix.ts";
import type {
  RecursiveArray,
  Size,
  Size2D,
  TensorLike,
  TypedArray,
} from "./types.ts";

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

export const zeros = (size: number): Float32Array => new Float32Array(size);
export const zeros2D = (width: number, height: number): Float32Array[] => {
  const result: Float32Array[] = new Array(height);
  for (let y = 0; y < height; y++) {
    result[y] = zeros(width);
  }
  return result;
};
export const zeros3D = (
  width: number,
  height: number,
  depth: number,
): Float32Array[][] => {
  const result: Float32Array[][] = new Array(depth);
  for (let z = 0; z < depth; z++) {
    result[z] = zeros2D(width, height);
  }
  return result;
};
export const ones = (size: number): Float32Array =>
  new Float32Array(size).fill(1);
export const ones2D = (width: number, height: number): Float32Array[] => {
  const result = new Array(height);
  for (let y = 0; y < height; y++) {
    result[y] = ones(width);
  }
  return result;
};
export const values = (size: number, value: number): Float32Array =>
  new Float32Array(size).fill(value);
export const values2D = (
  width: number,
  height: number,
  value: number,
): Float32Array[] => {
  const result: Float32Array[] = new Array(height);
  for (let y = 0; y < height; y++) {
    result[y] = values(width, value);
  }
  return result;
};
export const values3D = (
  width: number,
  height: number,
  depth: number,
  value: number,
): Float32Array[][] => {
  const result: Float32Array[][] = new Array(depth);
  for (let z = 0; z < depth; z++) {
    result[z] = values2D(width, height, value);
  }
  return result;
};
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

export function to1D(size: Size): number {
  const size2d = (size as Size2D);
  if (size2d.y) {
    return size2d.x * size2d.y;
  } else {
    return size as number;
  }
}

export function to2D(size: Size = 1): Size2D {
  return Number(size)
    ? { x: size as number, y: size as number } as Size2D
    : size as Size2D;
}

export function iterate2D(
  mat: { x: number; y: number } | CPUMatrix,
  callback: (i: number, j: number) => void,
): void {
  for (let i = 0; i < mat.x; i++) {
    for (let j = 0; j < mat.y; j++) {
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

export function flatten<
  T extends number | Promise<number> | TypedArray,
>(
  arr: T | RecursiveArray<T>,
  result: T[] = [],
  skipTypedArray = false,
): T[] | ArrayBufferLike {
  if (result == null) {
    result = [];
  }
  if (Array.isArray(arr) || isTypedArray(arr) && !skipTypedArray) {
    for (let i = 0; i < arr.length; ++i) {
      flatten(arr[i], result, skipTypedArray);
    }
  } else {
    result.push(arr as T);
  }
  return result;
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
      `[${shape}] does not match the input size ${a.length}${
        isComplex ? " for a complex tensor" : ""
      }.`,
    );
  }

  return createNestedArray(0, shape, a, isComplex);
}

export const average = (...args: number[]) =>
  args.reduce((a, b) => a + b) / args.length;

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
