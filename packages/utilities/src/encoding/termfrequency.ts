import { Matrix } from "../mod.ts";
import type { DataType } from "../utils/common_types.ts";

/**
 * Convert 2D array of indices into multi-hot encoded vectors
 * where each index contains the number of times the respective
 * value appears in a sample (term frequency encoder).
 */
export class TfEncoder {
  /** Size of encoded vectors. */
  mappingSize: number;
  constructor(mappingSize: number) {
    this.mappingSize = mappingSize;
  }
  /** Encoding values into count vectors */
  transform<DT extends DataType>(targets: Matrix<DT>): Matrix<DT>;
  transform<DT extends DataType>(targets: number[][], dType: DT): Matrix<DT>;
  transform<DT extends DataType>(
    targets: number[][] | Matrix<DT>,
    dType?: DT
  ): Matrix<DT> {
    if (!dType && !(targets instanceof Matrix))
      throw new Error("dType required when not dealing with matrices.");
    const dataType = dType || (targets as Matrix<DT>).dType;
    const res = new Matrix<DT>(dataType, [targets.length, this.mappingSize]);
    let i = 0;
    while (i < targets.length) {
      const row = targets instanceof Matrix ? targets.row(i) : targets[i];
      let j = 0;
      while (j < row.length) {
        if (Number(row[j]) >= row.length) {
            j += 1;
            continue;
        }
        res.setAdd(i, Number(row[j]), 1);
        j += 1;
      }

      i += 1;
    }
    return res;
  }
}
