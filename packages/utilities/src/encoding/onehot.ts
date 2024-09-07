import { Matrix, type MatrixLike } from "../mod.ts";
import type { DataType } from "../utils/common_types.ts";

/**
 * Convert an array of indices into one-hot encoded vectors.
 */
export class OneHotEncoder {
  /** Size of one-hot encoded vectors. */
  mappingSize: number;
  constructor(mappingSize: number) {
    this.mappingSize = mappingSize;
  }
  /** One-hot encoding of values */
  transform<DT extends DataType>(targets: number[], dType: DT): Matrix<DT> {
    const res = new Matrix<DT>(dType, [targets.length, this.mappingSize]);
    let i = 0;
    while (i < targets.length) {
      const index = targets[i];
      if (index >= this.mappingSize) {
        i += 1;
        continue;
      }
      res.setCell(i, index, 1);
      i += 1;
    }
    return res;
  }
  untransform<DT extends DataType>(data: MatrixLike<DT>): number[] {
    const matrix = new Matrix(data);
    const res = new Array(matrix.nRows);
    for (let i = 0; i < res.length; i += 1) {
      const idx = matrix.row(i).findIndex((x) => x === 1);
      res[i] = idx;
    }
    return res;
  }
}
