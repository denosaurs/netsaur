import type { DataType } from "./common_types.ts";
import { Matrix, type MatrixLike } from "./misc/matrix.ts";

/** A very basic, low-effort multiplication. */
export function multiplyDiags<T extends DataType>(
  x: MatrixLike<T>,
  y: ArrayLike<number>,
): Matrix<T> {
  const res = new Matrix(x);
  if (y.length !== res.nCols) {
    throw new Error(
      `Expected diagonal vector of shape (${res.nCols}, 1). Found (${y.length}, 1).`,
    );
  }
  let i = 0;
  while (i < res.nRows) {
    const offset = i * res.nCols;
    let j = 0;
    while (j < y.length) {
      res.setCell(
        i,
        j,
        // @ts-ignore types will always match
        x.data[offset + j] * (typeof res[0] === "bigint" ? BigInt(y[j]) : y[j]),
      );
      j += 1;
    }
    i += 1;
  }
  return res as Matrix<T>;
}
