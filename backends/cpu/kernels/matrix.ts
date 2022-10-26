import { cpuZeroes2D } from "../../../core/tensor.ts";
import { CPUTensor, Rank } from "../../../core/types.ts";
import { iterate1D, iterate2D } from "../../../core/util.ts";

export class CPUMatrix {
  static dot(matA: CPUTensor<Rank.R2>, matB: CPUTensor<Rank.R2>) {
    const res = cpuZeroes2D([matB.x, matA.y]);
    iterate2D([matB.x, matA.y], (x: number, y: number) => {
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

  static transpose(mat: CPUTensor<Rank.R2>) {
    const res = cpuZeroes2D([mat.y, mat.x]);
    iterate2D(mat.to2D(), (x: number, y: number) => {
      const i = x + y * mat.x;
      const j = y + x * mat.y;
      res.data[j] = mat.data[i];
    });
    return res;
  }
}
