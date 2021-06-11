import { Shape } from "./backend.ts";

export class CPUBackend {
  public async matMul(
    a: number[],
    b: number[],
    shapeA: Shape,
    shapeB: Shape,
  ): Promise<Float32Array> {
    const aNumRows = shapeA[0],
      aNumCols = shapeA[1],
      bNumCols = shapeB[1];

    const shape: Shape = [aNumRows, bNumCols];
    const m = new Float32Array(aNumRows * bNumCols);

    for (let r = 0; r < aNumRows; ++r) {
      for (let c = 0; c < bNumCols; ++c) {
        const cell = bNumCols * r + c;
        m[cell] = 0;
        for (let i = 0; i < aNumCols; ++i) {
          m[cell] += a[aNumCols * r + i] * b[bNumCols * i + c];
        }
      }
    }

    return m;
  }

  // static vecDotMul(
  //   a: Float32Array,
  //   b: Float32Array,
  // ): Float32Array {
  //   const res: Float32Array = new Float32Array(a.length);
  //   for (let i = 0; i < a.length; i++) {
  //     res[i] = a[i] * b[i + 1];
  //   }
  //   return res;
  // }
}
