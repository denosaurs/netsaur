import { Matrix, type MatrixLike } from "../mod.ts";
import type { DataType, DType, DTypeValue } from "../utils/common_types.ts";

/** Convert class labels into categorical variables (one-hot) */
export class CategoricalEncoder<T> {
  /** Map categories to indices */
  mapping: Map<T, number>;
  /** An internal counter for remembering the last index in mapping. */
  #lastToken: Uint32Array;
  constructor() {
    this.mapping = new Map();
    this.#lastToken = new Uint32Array(1);
  }
  /** Construct a mapping from a given set of text. */
  fit(targets: T[]): this {
    let i = 0;
    while (i < targets.length) {
      if (!this.mapping.has(targets[i])) {
        const token = this.#incrementToken();
        this.mapping.set(targets[i], token);
      }
      i += 1;
    }
    return this;
  }
  /** One-hot encoding of categorical values */
  transform<DT extends DataType>(targets: T[], dType: DT): Matrix<DT> {
    const res = new Matrix<DT>(dType, [targets.length, this.#lastToken[0]]);
    let i = 0;
    while (i < targets.length) {
      const index = this.mapping.get(targets[i]);
      if (index !== 0 && !index) {
        i += 1;
        continue;
      }
      res.setCell(i, index, 1);
      i += 1;
    }
    return res;
  }
  untransform<DT extends DataType>(data: MatrixLike<DT>): T[] {
    const matrix = new Matrix(data);
    const res = new Array(matrix.nRows);
    for (let i = 0; i < res.length; i += 1) {
      const idx = matrix.row(i).findIndex((x) => x === 1);
      res[i] = this.getOg(idx) || "__unknown__";
    }
    return res;
  }
  getOg(data: number): T | undefined {
    for (const [k, v] of this.mapping.entries()) {
      if (v === data) {
        return k;
      }
    }
    return undefined;
  }
  #incrementToken(): number {
    return Atomics.add(this.#lastToken, 0, 1);
  }
  /**
   * Convert softmax outputs into categorical outputs
   * This method mutates the original matrix.
   * @returns The modified matrix.
   */
  static fromSoftmax<DT extends DataType>(data: MatrixLike<DT>): Matrix<DT> {
    const matrix = new Matrix(data);
    for (let i = 0; i < matrix.nRows; i += 1) {
      const max = matrix
        .row(i)
        // @ts-ignore It can reduce.
        .reduce(
          (acc: number, curr: DTypeValue<DT>, i: number, arr: DType<DT>) =>
            arr[acc] > curr ? acc : i,
          0,
        );
      if (
        data.data instanceof BigInt64Array ||
        data.data instanceof BigUint64Array
      ) {
        const newR = new Array(matrix.nCols).fill(0n);
        newR[max] = 1n;
        matrix.setRow(i, newR);
      } else {
        const newR = new Array(matrix.nCols).fill(0);
        newR[max] = 1;
        matrix.setRow(i, newR);
      }
    }
    return matrix;
  }
}
