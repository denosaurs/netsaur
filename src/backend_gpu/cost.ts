import { GPUTensor, Rank, Shape } from "../../core/types.ts";
import { WebGPUBackend } from "../../deps.ts";
import { gpuZeroes } from "../../mod.ts";
import { cost, mse_prime, prime } from "./kernels/cost.ts";
import { mse } from "./kernels/cost.ts";

export class GPUCostFunction {
  name!: string;
  dInput!: GPUTensor<Rank>;
  error!: GPUTensor<Rank>;
  buffer!: GPUBuffer;
  protected backend!: WebGPUBackend;
  constructor(backend: WebGPUBackend) {
    this.backend = backend;
  }
  reset(shape: Shape[Rank]) {
    if (shape.at(-1) != this.dInput.shape.at(-1)) {
      this.dInput = gpuZeroes(shape);
    }
  }
  initialize(shape: Shape[Rank]) {
    this.dInput = gpuZeroes(shape);
    this.error = gpuZeroes([4]);
    const usage = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;
    this.buffer = this.backend.device.createBuffer({ size: 4, usage });
  }
  // dont block computations when copying output
  async getOutput() {
    const commandEncoder = this.backend.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.error.data.buffer,
      0,
      this.buffer,
      0,
      4,
    );
    this.backend.device.queue.submit([commandEncoder.finish()]);
    await this.buffer.mapAsync(GPUMapMode.READ);
    return new Float32Array(this.buffer.getMappedRange().slice(0));
  }
  async cost(_y: GPUTensor<Rank>, _yHat: GPUTensor<Rank>) {}
  async prime(_y: GPUTensor<Rank>, _yHat: GPUTensor<Rank>) {}
}

export class MSE extends GPUCostFunction {
  name = "mse";
  async cost(y: GPUTensor<Rank>, yHat: GPUTensor<Rank>) {
    await cost(this.backend, y, yHat, this.error, mse);
  }

  async prime(y: GPUTensor<Rank>, yHat: GPUTensor<Rank>) {
    await prime(this.backend, y, yHat, this.dInput, mse_prime);
  }
}
