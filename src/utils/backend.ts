import { WebGPUBackend } from "../backend/wgpu.ts";
import { CPUBackend } from "../backend/cpu.ts";

export type Shape = [number, number];

export interface Backend {
  matMul(
    a: number[],
    b: number[],
    shapeA: Shape,
    shapeB: Shape,
  ): Promise<Float32Array>;
}

export const dotProduct = (a: Array<number>, b: Array<number>) => {
  return a.map((ae, ai) => ae * b[ai]).reduce((p, c) => p + c);
};

export const randomWeight = (): number => {
  return Math.random() * 0.4 - 0.2;
};

export const values = (size: number, value: number): Float32Array => {
  return new Float32Array(size).fill(value);
};

// Unit testing
if (import.meta.main) {
  const backendGPU = await WebGPUBackend.init();
  const backendCPU = new CPUBackend();

  const shapeA: Shape = [1000, 1000];
  const shapeB: Shape = shapeA;
  const a = Array(shapeA[0] * shapeA[1]).fill(0).map((_, i) => i % 5);
  const b = a;

  const gpuStart = performance.now();
  const gpuRes = await backendGPU.matMul(a, b, shapeA, shapeB);
  const gpuEnd = performance.now();
  console.log(
    `GPU (${((gpuEnd - gpuStart) / 1000).toFixed(2)} seconds)`,
    gpuRes,
  );

  const cpuStart = performance.now();
  const cpuRes = await backendCPU.matMul(a, b, shapeA, shapeB);
  const cpuEnd = performance.now();
  console.log(
    `CPU (${((cpuEnd - cpuStart) / 1000).toFixed(2)} seconds)`,
    cpuRes,
  );
}
