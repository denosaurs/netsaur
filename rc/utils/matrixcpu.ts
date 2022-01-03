import { values } from './mod.ts';

export class MatrixCPU {
	static matrix(
    width: number,
    height: number,
    value: number,
  ): Float32Array[] {
  	const result: Float32Array[] = new Array(height);
    for (let y = 0; y < height; y++) {
        result[y] = values(width, value);
  	}
    return result;
  }

	static mat2d(
    a: Float32Array[]
  ): Float32Array {
    const res: Float32Array[] = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
			res.concat(a[i])
    }
    return res;
  }

  static matAdd(
    a: Float32Array[],
    b: Float32Array[],
  ): Float32Array[] {
    const res: Float32Array[] = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[i].length; j++) {
        res[i][j] = a[i][j] + b[i][j];
      }
    }
    return res;
  }

	static matMul(
    a: Float32Array[],
    b: number,
  ): Float32Array[] {
    const res: Float32Array[] = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[i].length; j++) {
        res[i][j] = a[i][j] * b;
      }
    }
    return res;
  }
	
  static matDotMul(
    a: Float32Array[],
    b: Float32Array[],
  ): Float32Array[] {
    const res: Float32Array[] = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[i].length; j++) {
        res[i][j] = a[i][j] * b[i][j];
      }
    }
    return res;
  }

	static vecDotMul(
    a: Float32Array,
    b: Float32Array,
  ): Float32Array {
    const res: Float32Array = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      res[i] = a[i] * b[i + 1];
    }
    return res;
  }
}
