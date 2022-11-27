import { WebGPUBackend } from "../../../deps.ts";
import { GPUTensor, LayerJSON, Rank, Shape } from "../../../core/types.ts";
import { backpropagate, feedforward } from "../kernels/activation.ts";
import { gpuZeroes } from "../../../mod.ts";
import { sigmoid, sigmoid_prime } from "../kernels/activation.ts";

/**
 * Activation Layer
 */
export class ActivationGPULayer {
  type!: string;
  outputSize!: Shape[Rank];
  input!: GPUTensor<Rank>;
  output!: GPUTensor<Rank>;
  dInputs!: GPUTensor<Rank>;
  protected backend: WebGPUBackend;
  constructor(backend: WebGPUBackend) {
    this.backend = backend;
  }
  reset(shape: Shape[Rank]) {
    if (shape.at(-1) != this.output.shape.at(-1)) {
      this.output = gpuZeroes(shape);
      this.dInputs = gpuZeroes(shape);
    }
    return shape
  }
  initialize(shape: Shape[Rank]) {
    this.output = gpuZeroes(shape);
    this.dInputs = gpuZeroes(shape);
    return shape
  }
  // deno-lint-ignore require-await
  async toJSON(): Promise<LayerJSON> {
    return {
      outputSize: this.outputSize,
      type: this.type,
    };
  }
  static fromJSON(_: LayerJSON, backend: WebGPUBackend) {
    return new ActivationGPULayer(backend);
  }
}

export class SigmoidGPULayer extends ActivationGPULayer {
  async feedForward(input: GPUTensor<Rank>) {
    this.input = input;
    await feedforward(this.backend, input, this.output, sigmoid);
    return this.output;
  }

  async backPropagate(dError: GPUTensor<Rank>) {
    await backpropagate(
      this.backend,
      this.output,
      dError,
      this.dInputs,
      sigmoid_prime,
    );
    return this.dInputs;
  }
}
