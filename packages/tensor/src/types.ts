export type DataType =
  | "u8"
  | "u16"
  | "u32"
  | "u64"
  | "i8"
  | "i16"
  | "i32"
  | "i64"
  | "f32"
  | "f64";

/**
 * Rank Types.
 */
export enum Rank {
  /**
   * Scalar   (magnitude only).
   */
  R1 = 1,

  /**
   * Vector   (magnitude and direction).
   */
  R2 = 2,

  /**
   * Matrix   (table of numbers).
   */
  R3 = 3,

  /**
   *  3-Tensor (cube of numbers)
   */
  R4 = 4,

  /**
   * Rank 5 Tensor
   */
  R5 = 5,

  /**
   * Rank 6 Tensor
   */
  R6 = 6,
}
export interface TypedArrayMapping {
  u8: Uint8Array;
  u16: Uint16Array;
  u32: Uint32Array;
  u64: BigUint64Array;
  i8: Int8Array;
  i16: Int16Array;
  i32: Int32Array;
  i64: BigInt64Array;
  f32: Float32Array;
  f64: Float64Array;
}

export interface TypedArrayConstructorMapping {
  u8: Uint8ArrayConstructor;
  u16: Uint16ArrayConstructor;
  u32: Uint32ArrayConstructor;
  u64: BigUint64ArrayConstructor;
  i8: Int8ArrayConstructor;
  i16: Int16ArrayConstructor;
  i32: Int32ArrayConstructor;
  i64: BigInt64ArrayConstructor;
  f32: Float32ArrayConstructor;
  f64: Float64ArrayConstructor;
}

interface TypedArrayValueMapping {
  u8: number;
  u16: number;
  u32: number;
  u64: bigint;
  i8: number;
  i16: number;
  i32: number;
  i64: bigint;
  f32: number;
  f64: number;
}

export type DTypeValue<T extends keyof TypedArrayValueMapping> = T extends
  keyof TypedArrayValueMapping ? TypedArrayValueMapping[T] : never;

type AddableTypes = number | bigint;

export type AddDTypeValues<
  T1 extends AddableTypes,
  T2 extends AddableTypes,
> = T1 extends number ? T2 extends number ? number
  : T2 extends bigint ? bigint
  : never
  : T1 extends bigint ? T2 extends number ? bigint
    : T2 extends bigint ? bigint
    : never
  : never;

export type DType<T extends keyof TypedArrayMapping> = T extends
  keyof TypedArrayMapping ? TypedArrayMapping[T] : never;

export type DTypeConstructor<T extends keyof TypedArrayConstructorMapping> =
  T extends keyof TypedArrayConstructorMapping ? TypedArrayConstructorMapping[T]
    : never;

export type TypedArray =
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Float32Array
  | Float64Array;
