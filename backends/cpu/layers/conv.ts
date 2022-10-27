import {
  Activation,
  ConvLayerConfig,
  CPUTensor,
  LayerJSON,
  Rank,
  Shape,
  Shape2D,
} from "../../../core/types.ts";
import { iterate2D } from "../../../core/util.ts";
import { cpuZeroes2D, cpuZeroes3D, reshape3D, Tensor, toShape3D } from "../../../mod.ts";
import { CPUActivationFn, setActivation, Sigmoid } from "../activation.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Convolutional 2D layer.
 */
export class ConvCPULayer {
  config: ConvLayerConfig;
  outputSize!: Shape2D;
  padding: number;
  strides: Shape2D;
  activationFn: CPUActivationFn = new Sigmoid();

  input!: CPUTensor<Rank.R3>;
  kernel!: CPUTensor<Rank.R2>;
  padded!: CPUTensor<Rank.R3>;
  biases!: CPUTensor<Rank.R3>;
  output!: CPUTensor<Rank.R3>;

  constructor(config: ConvLayerConfig) {
    this.config = config;
    if (config.kernel) {
      this.kernel = new Tensor(config.kernel, config.kernelSize);
    } else {
      this.kernel = cpuZeroes2D(config.kernelSize);
    }
    this.padding = config.padding || 0;
    this.strides = config.strides || [2, 2];
    this.setActivation(config.activation ?? "linear");
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Shape[Rank]) {
    const size = toShape3D(inputSize);
    const wp = size[0] + 2 * this.padding;
    const hp = size[1] + 2 * this.padding;
    if (this.padding > 0) {
      const data = new Float32Array(wp * hp * size[2]).fill(255)
      this.padded = new Tensor(data, [wp, hp, size[2]]);
    }
    if (!this.config.kernel) {
      this.kernel.data = this.kernel.data.map(() => Math.random() * 2 - 1);
    }
    const wo = 1 + Math.floor((wp - this.kernel.x) / this.strides[0]);
    const ho = 1 + Math.floor((hp - this.kernel.y) / this.strides[1]);
    this.biases = cpuZeroes3D([wo, ho, size[2]]);
    if (!this.config.unbiased) {
      this.biases.data = this.biases.data.map(() => Math.random() * 2 - 1);
    }
    this.output = cpuZeroes3D([wo, ho, size[2]]);
    this.outputSize = [wo, ho];
  }

  setActivation(activation: Activation) {
    this.activationFn = setActivation(activation);
  }

  feedForward(inputTensor: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = reshape3D(inputTensor)
    if (this.padding > 0) {
      iterate2D([this.input.x, this.input.y], (i: number, j: number) => {
        const idx = this.padded.x * (this.padding + j) + this.padding + i;
        this.padded.data[idx] = this.input.data[j * this.input.x + i];
      });
    } else {
      this.padded = this.input;
    }
    iterate2D([this.output.x, this.output.y], (i: number, j: number) => {
      let sum = 0;
      iterate2D(this.kernel, (x: number, y: number) => {
        const k = this.padded.x * (j * this.strides[1] + y) +
          (i * this.strides[0] + x);
        const l = y * this.kernel.x + x;
        sum += this.padded.data[k] * this.kernel.data[l];
      });
      sum += this.biases.data[j * this.output.x + i];
      this.output.data[j * this.output.x + i] = this.activationFn.activate(sum);
    });
    return this.output;
  }

  backPropagate(errorTensor: CPUTensor<Rank>, rate: number) {
    const error = reshape3D(errorTensor)
    const cost = cpuZeroes3D(error.shape);
    for (const i in cost.data) {
      const activation = this.activationFn.prime(this.output.data[i]);
      cost.data[i] = error.data[i] * activation;
    }
    iterate2D(this.kernel, (i: number, j: number) => {
      let sum = 0;
      iterate2D([cost.x, cost.y], (x: number, y: number) => {
        const k = this.padded.x * (j * this.strides[1] + y) +
          (i * this.strides[0] + x);
        const l = y * cost.x + x;
        sum += this.padded.data[k] * cost.data[l];
      });
      const idx = j * this.kernel.x + i;
      this.kernel.data[idx] += this.activationFn.activate(sum) * rate;
    });
    //TODO: fix batches
    for (let i = 0; i < cost.data.length; i++) {
      this.biases.data[i] += cost.data[i];
    }
  }

  async toJSON() {
    return {
      outputSize: this.outputSize,
      type: "conv",
      kernel: await this.kernel.toJSON(),
      biases: await this.biases.toJSON(),
      strides: this.strides,
      padding: this.padding,
    };
  }

  static fromJSON(
    { outputSize, kernel, type, biases, strides, padding }: LayerJSON,
  ): ConvCPULayer {
    if (type !== "conv") {
      throw new Error(
        "Cannot cannot create a Convolutional layer from a" +
          type.charAt(0).toUpperCase() + type.slice(1) +
          "Layer",
      );
    }
    if (biases === undefined || kernel === undefined) {
      throw new Error("Layer imported must be initialized");
    }
    const layer = new ConvCPULayer({
      kernelSize: kernel.shape as Shape2D,
      padding,
      strides: strides as Shape2D,
    });
    layer.outputSize = outputSize as Shape2D;
    layer.kernel = Tensor.fromJSON(kernel);
    layer.biases = Tensor.fromJSON(biases);
    return layer;
  }
}
