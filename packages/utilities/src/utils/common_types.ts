import type { DataType, Matrix } from "../../../tensor/mod.ts";

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
