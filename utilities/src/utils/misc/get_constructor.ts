import type { DataType, DTypeConstructor } from "../common_types.ts";

export function getConstructor<DT extends DataType>(
  dType: DT,
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
