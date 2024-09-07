import type { Matrix } from "./mod.ts";

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

export type DTypeValue<T extends keyof TypedArrayValueMapping> =
  T extends keyof TypedArrayValueMapping ? TypedArrayValueMapping[T] : never;

type AddableTypes = number | bigint;

export type AddDTypeValues<
  T1 extends AddableTypes,
  T2 extends AddableTypes
> = T1 extends number
  ? T2 extends number
    ? number
    : T2 extends bigint
    ? bigint
    : never
  : T1 extends bigint
  ? T2 extends number
    ? bigint
    : T2 extends bigint
    ? bigint
    : never
  : never;

export type DType<T extends keyof TypedArrayMapping> =
  T extends keyof TypedArrayMapping ? TypedArrayMapping[T] : never;

export type DTypeConstructor<T extends keyof TypedArrayConstructorMapping> =
  T extends keyof TypedArrayConstructorMapping
    ? TypedArrayConstructorMapping[T]
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

export type Constructor<T> = new (length: number) => T;

export interface Sliceable {
  filter(
    predicate: (
      value: unknown,
      index: number,
      array: unknown[]
    ) => value is unknown
  ): Sliceable;
  slice(start?: number, end?: number): Sliceable;
  length: number;
}

export function getDataType<DT extends DataType>(data: DType<DT>): DT {
  return (
    data instanceof Uint8Array
      ? "u8"
      : data instanceof Uint16Array
      ? "u16"
      : data instanceof Uint32Array
      ? "u32"
      : data instanceof Int8Array
      ? "i8"
      : data instanceof Int16Array
      ? "i16"
      : data instanceof Int32Array
      ? "i32"
      : data instanceof Float32Array
      ? "f32"
      : data instanceof Float64Array
      ? "f64"
      : "u8"
  ) as DT; // shouldn't reach "u8"
}

export interface Image2d {
  /** Width of the image */
  width: number;
  /** Height of the image */
  height: number;
  /**
   * Number of channels in the image
   * For a regular RGBA image, the value
   * will be 4.
   */
  channels: number;
  /** Array of length width * height * channels */
  data: Uint8ClampedArray;
}

export interface Patch2d {
  /** Width of the patch */
  width: number;
  /** Height of the patch */
  height: number;
}
export interface PatchCollection extends Patch2d {
  /**
   * Number of channels in the image
   * For a regular RGBA image, the value
   * will be 4.
   */
  channels: number;
  /** Number of patches in the collection */
  size: number;
  data: Uint8ClampedArray;
}

export type Pixel = [number, number, number, number?];

export interface StandardizeConfig {
  /** Whether to convert everything to lowercase before fitting / transforming */
  lowercase?: boolean;
  /** Whether to strip HTML tags */
  stripHtml?: boolean;
  /** Whether to replace multiple whitespaces. */
  normalizeWhiteSpaces?: boolean;
  /** Strip Newlines */
  stripNewlines?: boolean;
  /** Remove stop words from text */
  removeStopWords?: "english" | false | string[];
}

export type VectorizerMode = "count" | "indices" | "multihot" | "tfidf";

export type VectorizerModeConfig =
  | {
      mode: "count";
      config?: Partial<BaseVectorizerOptions>;
    }
  | {
      mode: "indices";
      config?: Partial<BaseVectorizerOptions & { size: number }>;
    }
  | {
      mode: "multihot";
      config?: Partial<BaseVectorizerOptions>;
    }
  | {
      mode: "tfidf";
      config?: Partial<BaseVectorizerOptions & { idf: Float64Array }>;
    };

export interface TokenizerModeConfig {
  mode: "whitespace";
  config?: Partial<BaseVectorizerOptions>;
}

export interface BaseVectorizerOptions {
  /** Map words to indices */
  vocabulary: Map<string, number>;
  /** Options for standardizing text */
  standardize: StandardizeConfig | ((s: string) => string);
  /** Words to ignore from vocabulary */
  skipWords: "english" | false | string[];
}

export interface BaseTokenizerOptions {
  /** Map words to indices */
  vocabulary: Map<string, number>;
  /** Options for standardizing text */
  standardize: StandardizeConfig | ((s: string) => string);
  /** Words to ignore from vocabulary */
  skipWords: "english" | false | string[];
}

export interface Tokenizer {
  fit(text: string | string[]): unknown;
  transform(text: string | string[]): number[];
}

export interface Cleaner {
  clean(text: string): string;
  clean(text: string[]): string[];
}

export interface Vectorizer {
  transform<T extends DataType>(tokens: number[][], dType: T): Matrix<T>;
}

export interface Transformer {
  fit<T extends DataType>(data: Matrix<T>): Transformer;
  transform<T extends DataType>(data: Matrix<T>): Matrix<T>;
}
