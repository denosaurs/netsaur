import { DataType, DataTypeArray } from "../../deps.ts";
import { iterate1D, iterate2D } from "../../core/util.ts";
export class CPUMatrix<T extends DataType = DataType> {
  deltas: DataTypeArray<T>;
  constructor(
    public data: DataTypeArray<T>,
    public x: number,
    public y: number,
  ) {
    this.deltas = this.data;
  }

  static with(
    x: number,
    y: number,
  ) {
    const data = new Float32Array(x * y);
    return new this(data, x, y);
  }

  static add(matA: CPUMatrix, matB: CPUMatrix) {
    const res = CPUMatrix.with(matA.x, matA.y);
    iterate1D(matA.data.length, (i: number) => {
      res.data[i] = matA.data[i] + matB.data[i];
    });
    return res;
  }

  static dot(matA: CPUMatrix, matB: CPUMatrix) {
    const res = CPUMatrix.with(matB.x, matA.y);
    iterate2D({ x: matB.x, y: matA.y }, (x: number, y: number) => {
      let sum = 0;
      iterate1D(matA.x, (k: number) => {
        const a = k + y * matA.x;
        const b = x + k * matB.x;
        sum += matA.data[a] * matB.data[b];
      });
      const idx = x + y * matB.x;
      res.data[idx] = sum;
    });
    return res;
  }

  static Tdot(matA: CPUMatrix, matB: CPUMatrix) {
    const res = CPUMatrix.with(matB.y, matA.y);
    iterate2D({ x: matB.y, y: matA.y }, (x: number, y: number) => {
      let sum = 0;
      iterate1D(matA.x, (k: number) => {
        const a = k + y * matA.x;
        const b = k + x * matB.x;
        sum += matA.data[a] * matB.data[b];
      });
      const idx = x + y * matB.y;
      res.data[idx] = sum;
    });
    return res;
  }

  static sub(matA: CPUMatrix, matB: CPUMatrix) {
    const res = CPUMatrix.with(matA.x, matA.y);
    iterate1D(matA.data.length, (i: number) => {
      res.data[i] = matA.data[i] - matB.data[i];
    });
    return res;
  }

  static reduce(mat: CPUMatrix, func: (acc: number, val: number) => number) {
    iterate1D(mat.data.length, (i: number) => {
      mat.data[0] = func(mat.data[0], mat.data[i]);
    });
    return mat;
  }

  static mirror(mat: CPUMatrix) {
    iterate1D(mat.data.length, (i: number) => {
      mat.data[i] = mat.data[mat.data.length - i - 1];
    });
    return mat;
  }

  static transpose(mat: CPUMatrix) {
    const res = CPUMatrix.with(mat.y, mat.x);
    iterate2D(mat, (x: number, y: number) => {
      const i = x + y * mat.x;
      const j = y + x * mat.y;
      res.data[j] = mat.data[i];
    });
    return res;
  }

  fill(val: number) {
    this.data.fill(val);
  }

  getData(x: number, y: number) {
    const ix = this.y * x + y;
    if (ix < 0 || ix >= this.data.length) {
      throw new Error("get accessor is skewed");
    }
    return this.data[ix];
  }

  setData(x: number, y: number, val: number) {
    const ix = this.y * x + y;
    if (ix < 0 || ix >= this.data.length) {
      throw new Error("set accessor is skewed");
    }
    this.data[ix] = val;
  }

  getDelta(x: number, y: number) {
    const ix = this.y * x + y;
    if (ix < 0 || ix >= this.deltas.length) {
      throw new Error("get accessor is skewed");
    }
    return this.deltas[ix];
  }

  setDelta(x: number, y: number, val: number) {
    const ix = this.y * x + y;
    if (ix < 0 || ix >= this.deltas.length) {
      throw new Error("set accessor is skewed");
    }
    this.deltas[ix] = val;
  }

  toJSON() {
    return {
      data: this.data,
      x: this.x,
      y: this.y,
    };
  }
  fmt() {
    let res = "";
    iterate1D(this.y, (i: number) => {
      const row = this.data.slice(i * this.x, (i + 1) * this.x);
      iterate1D(row.length, (j: number) => {
        res += row[j].toString() + " ";
      });
      res += "\n";
    });
    return res;
  }
}
