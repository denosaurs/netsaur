// export type Matrix = {

// }
export { MatrixGPU } from './matrixgpu.ts';
export { Matrix } from './matrixcpu.ts';
export const dotProduct = (a: Array<number>, b: Array<number>) => {
  return a.map((ae, ai) => ae * b[ai]).reduce((p, c) => p + c);
}
export const randomWeight = (): number => {
  return Math.random() * 0.4 - 0.2;
}
export const values = (size: number, value: number): Float32Array => {
  return new Float32Array(size).fill(value);
}

export interface Backend {
  vecDotMul(
    a: Float32Array,
    b: Float32Array,
  ): Float32Array;

  matDotMul(
    a: Float32Array[],
    b: Float32Array[],
  ): Float32Array[];

  matMul(
    a: Float32Array[],
    b: number,
  ): Float32Array[];

  matAdd(
    a: Float32Array[],
    b: Float32Array[],
  ): Float32Array[];
}


