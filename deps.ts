export type {
  DataArray,
  DataType,
} from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/types.ts";
export { DataArrayConstructor } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/types.ts";
export { ensureType } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/util.ts";
export { WebGPUBackend } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/webgpu/backend.ts";
export { WebGPUData } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/webgpu/data.ts";

// import type { DataType } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/types.ts";

// export type DataArray<T extends DataType> = T extends "u32" ? Uint32Array : T extends "i32" ? Int32Array : T extends "f32" ? Float32Array : never
