import { CPUMatrix } from "../src/cpu/matrix.ts";

const matA = new CPUMatrix(
  new Float32Array([
    1,
    2,
    3,
    4,
    5,
    6,
  ]),
  3,
  2,
);

const matB = new CPUMatrix(
  new Float32Array([
    7,
    8,
    9,
    10,
    11,
    12,
  ]),
  2,
  3,
);

console.log(CPUMatrix.dot(matA, matB).data);
