import { WebGPUBackend } from "../../../deps.ts";
import {
  GPUTensor,
  LayerJSON,
  PoolLayerConfig,
  Rank,
  Shape,
  Shape2D,
  Shape3D,
} from "../../../core/types.ts";
import { feedforward_max } from "../kernels/pool.ts";
import { checkShape, checkTensor, gpuZeroes4D } from "../../../mod.ts";
import { InvalidPoolError } from "../../../core/error.ts";
import { GPUInstance } from "../mod.ts";

/**
 * Pool Layer
 */
export class PoolGPULayer {
  type = "pool";
  outputSize!: Shape3D;
  strides: Shape2D;
  mode: "max" | "avg";

  input!: GPUTensor<Rank.R4>;
  indices!: GPUTensor<Rank.R4>;
  output!: GPUTensor<Rank.R4>;
  dInputs!: GPUTensor<Rank.R4>;

  #backend: WebGPUBackend;

  constructor(config: PoolLayerConfig, backend: WebGPUBackend) {
    this.mode = config.mode || "max";
    this.strides = config.strides || [1, 1];
    this.#backend = backend;
  }

  reset(inputShape: Shape[Rank]) {
    const shape = checkShape(inputShape, Rank.R4);
    const [w, h, f] = this.outputSize;
    this.indices = gpuZeroes4D([w, h, f, shape[3]]);
    this.output = gpuZeroes4D([w, h, f, shape[3]]);
    return this.output.shape;
  }

  initialize(inputSize: Shape[Rank]) {
    const size = checkShape(inputSize, Rank.R4);
    if (
      size[0] % this.strides[0] || size[1] % this.strides[1] ||
      this.strides[0] == 1 || this.strides[1] == 1
    ) {
      throw new InvalidPoolError(size, this.strides);
    }
    const w = size[0] / this.strides[0];
    const h = size[1] / this.strides[1];
    this.outputSize = [w, h, size[2]];
    this.indices = gpuZeroes4D([w, h, size[2], size[3]]);
    this.output = gpuZeroes4D([w, h, size[2], size[3]]);
    return this.output.shape;
  }

  async feedForward(inputTensor: GPUTensor<Rank>) {
    this.input = checkTensor(inputTensor, Rank.R4);
    await feedforward_max(
      this.#backend,
      this.input,
      this.indices,
      this.output,
      this.strides,
    );
    return this.output;
  }

  // deno-lint-ignore require-await
  async backPropagate(_errorTensor: GPUTensor<Rank>, _rate: number) {
    // const dError = checkTensor(errorTensor, Rank.R2);
    // await backpropagate(
    //   this.#backend,
    //   this.input,
    //   this.weights,
    //   this.biases,
    //   dError,
    //   this.dInputs,
    //   rate,
    // );
    return this.dInputs;
  }

  toJSON() {
    return {
      outputSize: this.outputSize,
      type: this.type,
      strides: this.strides,
      mode: this.mode,
    };
  }

  static fromJSON(
    { outputSize, strides, mode }: LayerJSON,
  ): PoolGPULayer {
    const layer = new PoolGPULayer(
      { strides: strides as Shape2D, mode },
      GPUInstance.backend!,
    );
    layer.outputSize = outputSize as Shape3D;
    return layer;
  }
}
