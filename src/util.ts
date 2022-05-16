import { DataArray, DataArrayConstructor, DataType } from "../deps.ts";

export function getType<T extends DataType>(type: DataArray<T>) {
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
  ) as DataArrayConstructor<T>;
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
  ) as DataArrayConstructor<T>;
}

export class ActivationError extends Error {
  constructor(activation: string) {
    super(
      `Unknown activation function: ${activation}.  Available: "sigmoid", "tanh", "relu", "relu6" , "leakyrelu", "elu", "linear", "selu"`,
    );
  }
}
