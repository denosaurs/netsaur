import { DataArray, DataType } from "../deps.ts";

export function getType<T extends DataType>(type: DataArray<T>) {
    return (
        type instanceof Uint32Array ? "u32"
      : type instanceof Int32Array ? "i32"
      : type instanceof Float32Array ? "f32"
      : undefined
    )! as T
}