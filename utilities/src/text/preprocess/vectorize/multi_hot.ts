import type { DataType, DType } from "../../../utils/common_types.ts";
import { getConstructor } from "../../../utils/mod.ts";
import { Matrix } from "../../../mod.ts";

/**
 * Convert tokens into vectors based on term frequency
 */
export class MultiHotVectorizer {
  vocabSize: number;
  constructor(vocabSize: number) {
    this.vocabSize = vocabSize;
  }
  /**
   * Convert a document (string | array of strings) into vectors.
   */
  transform<T extends DataType>(tokens: number[][], dType: T): Matrix<T> {
    if (!this.vocabSize) {
      throw new Error("Vocab not initialized.");
    }
    const res = new Matrix(dType, [tokens.length, this.vocabSize]);
    let i = 0;
    while (i < tokens.length) {
      res.setRow(i, this.#transform<T>(tokens[i], dType));
      i += 1;
    }
    return res as Matrix<T>;
  }
  #transform<T extends DataType>(tokens: number[], dType: T): DType<T> {
    const res = new (getConstructor<T>(dType))(this.vocabSize);
    let i = 0;
    while (i < tokens.length) {
      if (tokens[i] < this.vocabSize) {
        res[tokens[i]] = typeof res[tokens[i]] === "bigint" ? 1n : 1;
      }
      i += 1;
    }
    return res as DType<T>;
  }
}
