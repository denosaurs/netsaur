import { dotProduct } from "./backend.ts";
export class Matrix {
  static transpose(a: any) {
    return a[0].map((x: any, i: any) => {
      return a.map((y: any, k: any) => {
        return y[i];
      });
    });
  }
  static dotMul(m1: any, m2: any) {
    return m1.map((x: Array<number>, i: any) => {
      return Matrix.transpose(m2).map((y: Array<number>, k: any) => {
        return dotProduct(x, y);
      });
    });
  }
}
