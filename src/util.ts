import { DataType, DataTypeArray, DataTypeArrayConstructor } from "../deps.ts";

export function getType<T extends DataType>(type: DataTypeArray<T>) {
  return (
    type instanceof Uint32Array
      ? "u32"
      : type instanceof Int32Array
      ? "i32"
      : type instanceof Float32Array
      ? "f32"
      : undefined
  )! as T;
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
