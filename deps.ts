export type {
  DataType,
  DataTypeArray,
  DataTypeArrayConstructor,
} from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/types/data.ts";
export {
  ensureDataType,
  lookupDataPrimitiveArrayConstructor,
} from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/util/data.ts";
export { WebGPUBackend } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/webgpu/backend.ts";
export { WebGPUData } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/webgpu/data.ts";

// import type { DataType } from "https://raw.githubusercontent.com/denosaurs/neo/main/backend/types.ts";

// export type DataTypeArray<T extends DataType> = T extends "u32" ? Uint32Array : T extends "i32" ? Int32Array : T extends "f32" ? Float32Array : never
