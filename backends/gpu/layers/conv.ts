import { WebGPUBackend } from "../../../deps.ts";
import {
  Activation,
  ConvLayerConfig,
  GPUTensor,
  InitFn,
  LayerJSON,
  Rank,
  Shape,
  Shape2D,
  Shape3D,
  Shape4D,
} from "../../../core/types.ts";
import { feedforward } from "../kernels/conv.ts";
import {
  checkShape,
  checkTensor,
  gpuTensor,
  gpuZeroes1D,
  gpuZeroes4D,
  Tensor,
} from "../../../mod.ts";
import { Uniform } from "../../../core/init.ts";
import { GPUInstance } from "../mod.ts";

/**
 * Conv Layer
 */
export class ConvGPULayer {
  type = "dense";
  config: ConvLayerConfig;
  outputSize!: Shape3D;
  kernelSize: Shape4D;
  paddedSize!: Shape3D;
  padding: number;
  strides: Shape2D;
  init: InitFn = new Uniform();

  input!: GPUTensor<Rank.R4>;
  weights!: GPUTensor<Rank.R4>;
  biases!: GPUTensor<Rank.R1>;
  output!: GPUTensor<Rank.R4>;
  dInputs!: GPUTensor<Rank.R4>;

  #backend: WebGPUBackend;

  constructor(config: ConvLayerConfig, backend: WebGPUBackend) {
    this.config = config;
    this.padding = config.padding || 0;
    this.strides = config.strides || [1, 1];
    this.kernelSize = config.kernelSize;
    this.#backend = backend;
  }

  reset(inputShape: Shape[Rank]) {
    return inputShape;
  }

  initialize(inputSize: Shape[Rank]) {
    const size = checkShape(inputSize, Rank.R4);
    const wp = size[0] + 2 * this.padding;
    const hp = size[1] + 2 * this.padding;
    this.paddedSize = [wp, hp, size[2]];
    const wo = 1 + Math.floor((wp - this.kernelSize[0]) / this.strides[0]);
    const ho = 1 + Math.floor((hp - this.kernelSize[1]) / this.strides[1]);
    this.outputSize = [wo, ho, this.kernelSize[3]];
    const inputShape = [size[0], size[1], size[2]] as Shape3D;
    this.weights = this.config.kernel
      ? gpuTensor(this.config.kernel, this.kernelSize)
      : this.init.init(inputShape, this.kernelSize, this.outputSize);
    this.biases = gpuZeroes1D([this.kernelSize[3]]);
    this.output = gpuZeroes4D([wo, ho, this.kernelSize[3], size[3]]);
    return this.outputSize
  }

  async feedForward(inputTensor: GPUTensor<Rank>) {
    this.input = checkTensor(inputTensor, Rank.R4);
    await feedforward(
      this.#backend,
      this.input,
      this.weights,
      this.biases,
      this.output,
      this.strides,
      this.padding
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

  async toJSON() {
    return {
      outputSize: this.outputSize,
      type: this.type,
      kernel: await this.weights.toJSON(),
      biases: await this.biases.toJSON(),
      strides: this.strides,
      paddedSize: this.paddedSize,
      padding: this.padding,
    };
  }

  static fromJSON(
    {
      outputSize,
      kernel,
      biases,
      strides,
      padding,
      paddedSize,
      activationFn,
    }: LayerJSON,
  ): ConvGPULayer {
    const layer = new ConvGPULayer({
      kernelSize: kernel!.shape as Shape4D,
      padding,
      strides: strides as Shape2D,
      activation: activationFn as Activation,
    }, GPUInstance.backend!);
    layer.paddedSize = paddedSize as Shape3D;
    layer.outputSize = outputSize as Shape3D;
    layer.weights = Tensor.fromJSON(kernel!);
    layer.biases = Tensor.fromJSON(biases!);
    return layer;
  }
}
