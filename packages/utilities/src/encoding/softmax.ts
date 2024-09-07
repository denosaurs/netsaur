import { Matrix, type MatrixLike } from "../mod.ts";
import type { DataType, DType, DTypeValue } from "../utils/common_types.ts";

/**
 * Convert a softmax output into one-hot vectors.
 * Mutates the input.
 */
export function transformSoftmaxMut<DT extends DataType>(
  targets: MatrixLike<DT>
): Matrix<DT> {
  const matrix = new Matrix(targets);
  for (let i = 0; i < matrix.nRows; i += 1) {
    const max = matrix
      .row(i)
      // @ts-ignore It can reduce.
      .reduce(
        (acc: number, curr: DTypeValue<DT>, i: number, arr: DType<DT>) =>
          arr[acc] > curr ? acc : i,
        0
      );
    if (
      targets.data instanceof BigInt64Array ||
      targets.data instanceof BigUint64Array
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
