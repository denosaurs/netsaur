import { WebGPUBackend } from "../../../deps.ts";
import {
  Activation,
  DenseLayerConfig,
  GPUTensor,
  InitFn,
  LayerJSON,
  Rank,
  Shape,
  Shape1D,
  Shape2D,
} from "../../../core/types.ts";
import { backpropagate, feedforward } from "../kernels/dense.ts";
import { checkShape, checkTensor, gpuZeroes2D, Tensor } from "../../../mod.ts";
import { Uniform } from "../../../core/init.ts";

/**
 * Regular Dense Layer
 */
export class DenseGPULayer {
  type = "dense";
  outputSize: Shape1D;
  init: InitFn = new Uniform();

  input!: GPUTensor<Rank.R2>;
  weights!: GPUTensor<Rank.R2>;
  biases!: GPUTensor<Rank.R2>;
  output!: GPUTensor<Rank.R2>;
  dInputs!: GPUTensor<Rank.R2>;

  #backend: WebGPUBackend;

  constructor(config: DenseLayerConfig, backend: WebGPUBackend) {
    this.outputSize = config.size;
    this.#backend = backend;
  }

  reset(inputShape: Shape[Rank]) {
    const shape = checkShape(inputShape, Rank.R2);
    if (shape[1] != this.output.y) {
      this.output = gpuZeroes2D([this.outputSize[0], shape[1]]);
      this.dInputs = gpuZeroes2D([this.weights.y, shape[1]]);
    }
    return [this.outputSize[0], shape[1]] as Shape2D;
  }

  initialize(inputShape: Shape[Rank]) {
    const shape = checkShape(inputShape, Rank.R2);
    const weights = [this.outputSize[0], shape[0]] as Shape2D;
    this.weights = this.init.init([shape[0]], weights, this.outputSize);
    this.biases = gpuZeroes2D([this.outputSize[0], 1]);
    this.output = gpuZeroes2D([this.outputSize[0], shape[1]]);
    this.dInputs = gpuZeroes2D(shape);
    return [this.outputSize[0], shape[1]] as Shape2D;
  }

  async feedForward(inputTensor: GPUTensor<Rank>) {
    this.input = checkTensor(inputTensor, Rank.R2);
    await feedforward(
      this.#backend,
      this.input,
      this.weights,
      this.biases,
      this.output,
    );
    return this.output;
  }

  async backPropagate(errorTensor: GPUTensor<Rank>, rate: number) {
    const dError = checkTensor(errorTensor, Rank.R2);
    await backpropagate(
      this.#backend,
      this.input,
      this.weights,
      this.biases,
      dError,
      this.dInputs,
      rate,
    );
    return this.dInputs;
  }

  async toJSON(): Promise<LayerJSON> {
    return {
      outputSize: this.outputSize,
      type: "dense",
      weights: await this.weights.toJSON(),
      biases: await this.biases.toJSON(),
    };
  }

  static fromJSON(
    { outputSize, activationFn, weights, biases }: LayerJSON,
    backend: WebGPUBackend,
  ) {
    const layer = new DenseGPULayer({
      size: outputSize as Shape1D,
      activation: activationFn as Activation,
    }, backend);
    layer.weights = Tensor.fromJSON(weights!);
    layer.biases = Tensor.fromJSON(biases!);
    return layer;
  }
}
