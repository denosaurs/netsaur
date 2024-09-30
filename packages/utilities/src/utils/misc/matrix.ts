/**
 * A 2D Tensor with more methods.
 * @module
 */

import type {
  AddDTypeValues,
  DataType,
  DType,
  DTypeConstructor,
  DTypeValue,
  Sliceable,
} from "../common_types.ts";
import { type NDArray, type Shape, Tensor, type TensorLike } from "./tensor.ts";

/** The base type implemented by Matrix */
export type MatrixLike<DT extends DataType> = {
  /** Raw 1D TypedArray supplied */
  data: DType<DT>;
  /** Number of rows, columns */
  shape: Shape<2>;
};

/**
 * Class for 2D Arrays.
 * This is not akin to a mathematical Matrix (a collection of column vectors).
 * This is a collection of row vectors.
 * A special case of Tensor for 2D data.
 */
export class Matrix<DT extends DataType>
  extends Tensor<DT, 2>
  implements Sliceable, MatrixLike<DT>
{
  /**
   * Create a matrix from a typed array
   * @param data Data to move into the matrix.
   * @param shape [rows, columns] of the matrix.
   */
  constructor(matrix: TensorLike<DT, 2>);
  constructor(array: NDArray<DT>[2], dType: DT);
  constructor(data: DType<DT>, shape: Shape<2>);
  constructor(dType: DT, shape: Shape<2>);
  constructor(
    data: NDArray<DT>[2] | DType<DT> | DT | TensorLike<DT, 2>,
    shape?: Shape<2> | DT
  ) {
    // @ts-ignore This call will work
    super(data, shape);
  }
  get head() {
    return this.slice(0, Math.min(this.nRows, 10));
  }
  get tail() {
    return this.slice(Math.max(this.nRows - 10, 0), this.nRows);
  }
  /** Convert the Matrix into a HTML table */
  get html(): string {
    let res = "<table>\n";
    res += "<thead><tr><DTh>idx</th>";
    for (let i = 0; i < this.nCols; i += 1) {
      res += `<DTh>${i}</th>`;
    }
    res += "</tr></thead>";
    let j = 0;
    for (const row of this.rows()) {
      res += `<tr><td><strong>${j}</strong></td>`;
      j += 1;
      for (const x of row) {
        res += `<td>${x}</td>`;
      }
      res += "</tr>";
    }
    res += "</table>";
    return res;
  }
  override get length(): number {
    return this.nRows;
  }
  /** Returns number of cols */
  get nCols(): number {
    return this.shape[1];
  }
  /** Returns number of rows */
  get nRows(): number {
    return this.shape[0];
  }
  /** Get the transpose of the matrix. This method clones the matrix. */
  get T(): Matrix<DT> {
    const resArr = new (this.data.constructor as DTypeConstructor<DT>)(
      this.nRows * this.nCols
    ) as DType<DT>;
    let i = 0;
    for (const col of this.cols()) {
      // @ts-ignore This line will work
      resArr.set(col, i * this.nRows);
      i += 1;
    }
    return new Matrix(resArr, this.shape);
  }
  /** Get a pretty version for printing. DO NOT USE FOR MATRICES WITH MANY COLUMNS. */
  get pretty(): string {
    let res = "";
    for (const row of this.rows()) {
      res += row.join("\t");
      res += "\n";
    }
    return res;
  }
  /** Alias for row */
  at(pos: number): DType<DT> {
    return this.row(pos);
  }
  /** Get the nth column in the matrix */
  col(n: number): DType<DT> {
    let i = 0;
    const col = new (this.data.constructor as DTypeConstructor<DT>)(
      this.nRows
    ) as DType<DT>;
    let offset = 0;
    while (i < this.nRows) {
      col[i] = this.data[offset + n];
      i += 1;
      offset += this.nCols;
    }
    return col;
  }
  colMean(): DType<DT> {
    const sum = this.colSum();
    let i = 0;
    const divisor = (
      typeof this.data[0] === "bigint" ? BigInt(this.nCols) : this.nCols
    ) as DTypeValue<DT>;
    while (i < sum.length) {
      sum[i] = (sum[i] as DTypeValue<DT>) / divisor;
      i += 1;
    }
    return sum;
  }
  /** Get a column array of all column sums in the matrix */
  colSum(): DType<DT> {
    const sum = new (this.data.constructor as DTypeConstructor<DT>)(
      this.nRows
    ) as DType<DT>;
    let i = 0;
    while (i < this.nCols) {
      let j = 0;
      while (j < this.nRows) {
        // @ts-ignore I'll fix this later
        sum[j] = (sum[j] + this.item(j, i)) as AddDTypeValues<
          DTypeValue<DT>,
          DTypeValue<DT>
        >;
        j += 1;
      }
      i += 1;
    }
    return sum;
  }
  /** Get the dot product of two matrices */
  dot(rhs: Matrix<DT>): number | bigint {
    if (rhs.nRows !== this.nRows) {
      throw new Error("Matrices must have equal rows.");
    }
    if (rhs.nCols !== this.nCols) {
      throw new Error("Matrices must have equal cols.");
    }
    let res = (typeof this.data[0] === "bigint" ? 0n : 0) as DTypeValue<DT>;
    let j = 0;
    while (j < this.nCols) {
      let i = 0;
      while (i < this.nRows) {
        const adder =
          (this.item(i, j) as DTypeValue<DT>) *
          (rhs.item(i, j) as DTypeValue<DT>);
        // @ts-ignore I'll fix this later
        res += adder as DTypeValue<DT>;
        i += 1;
      }
      j += 1;
    }
    return res;
  }
  /** Filter the matrix by rows */
  override filter(
    fn: (value: DType<DT>, row: number, _: DType<DT>[]) => boolean
  ): Matrix<DT> {
    const satisfying: number[] = [];
    let i = 0;
    while (i < this.nRows) {
      if (fn(this.row(i), i, [])) {
        satisfying.push(i);
      }
      i += 1;
    }
    const matrix = new Matrix(this.dType, [satisfying.length, this.nCols]);
    i = 0;
    while (i < satisfying.length) {
      // @ts-ignore This line will work
      matrix.setRow(i, this.row(satisfying[i]));
      i += 1;
    }
    return matrix;
  }
  /** Get an item using a row and column index */
  override item(row: number, col: number): DTypeValue<DT> {
    return this.data[row * this.nCols + col] as DTypeValue<DT>;
  }
  /** Get the nth row in the matrix */
  row(n: number): DType<DT> {
    return this.data.slice(n * this.nCols, (n + 1) * this.nCols) as DType<DT>;
  }
  rowMean(): DType<DT> {
    const sum = this.rowSum();
    let i = 0;
    const divisor = (
      typeof this.data[0] === "bigint" ? BigInt(this.nRows) : this.nRows
    ) as DTypeValue<DT>;
    while (i < sum.length) {
      sum[i] = (sum[i] as DTypeValue<DT>) / divisor;
      i += 1;
    }
    return sum;
  }
  /** Compute the sum of all rows */
  rowSum(): DType<DT> {
    const sum = new (this.data.constructor as DTypeConstructor<DT>)(
      this.nCols
    ) as DType<DT>;
    let i = 0;
    let offset = 0;
    while (i < this.nRows) {
      let j = 0;
      while (j < this.nCols) {
        // @ts-ignore This line will work
        sum[j] += this.data[offset + j];
        j += 1;
      }
      i += 1;
      offset += this.nCols;
    }
    return sum;
  }
  /**
   * Add a value to an existing element
   * Will throw an error if the types mismatch
   */
  setAdd(row: number, col: number, val: number | bigint) {
    // @ts-expect-error Must provide appropriate number/bigint argument
    this.data[row * this.nCols + col] += val;
  }
  /** Replace a column */
  setCol(col: number, val: ArrayLike<number>): number {
    let i = 0;
    while (i < this.nRows) {
      this.data[i * this.nCols + col] = val[i];
      i += 1;
    }
    return col;
  }
  /** Set a value in the matrix */
  setCell(row: number, col: number, val: number) {
    this.data[row * this.nCols + col] = val;
  }
  /** Replace a row */
  setRow(row: number, val: ArrayLike<number> | ArrayLike<bigint>) {
    // @ts-expect-error Must provide appropriate number/bigint argument
    this.data.set(val, row * this.nCols);
  }
  /** Slice matrix by rows */
  override slice(start = 0, end?: number): Matrix<DT> {
    return new Matrix<DT>(
      this.data.slice(
        start ? start * this.nCols : 0,
        end ? end * this.nCols : undefined
      ) as DType<DT>,
      [end ? end - start : this.nRows - start, this.nCols]
    );
  }
  /** Iterate through rows */
  *rows(): Generator<DType<DT>> {
    let i = 0;
    while (i < this.nRows) {
      yield this.data.slice(i * this.nCols, (i + 1) * this.nCols) as DType<DT>;
      i += 1;
    }
  }
  /** Iterate through columns */
  *cols(): Generator<DType<DT>> {
    let i = 0;
    while (i < this.nCols) {
      let j = 0;
      const col = new (this.data.constructor as DTypeConstructor<DT>)(
        this.nRows
      ) as DType<DT>;
      while (j < this.nRows) {
        col[j] = this.data[j * this.nCols + i];
        j += 1;
      }
      yield col;
      i += 1;
    }
  }

  [Symbol.for("Jupyter.display")](): Record<string, string> {
    return {
      // Plain text content
      "text/plain": this.pretty,

      // HTML output
      "text/html": this.html,
    };
  }
}
