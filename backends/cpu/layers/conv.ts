import {
  Activation,
  ConvLayerConfig,
  CPUTensor,
  LayerJSON,
  Rank,
  Shape,
  Shape2D,
} from "../../../core/types.ts";
import { iterate1D, iterate2D } from "../../../core/util.ts";
import {
  cpuZeroes2D,
  cpuZeroes3D,
  reshape3D,
  Tensor,
  toShape3D,
} from "../../../mod.ts";
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
  biases!: CPUTensor<Rank.R2>;
  error!: CPUTensor<Rank.R3>;
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
      const data = new Float32Array(wp * hp * size[2]).fill(255);
      this.padded = new Tensor(data, [wp, hp, size[2]]);
    }
    if (!this.config.kernel) {
      this.kernel.data = this.kernel.data.map(() => Math.random() * 2 - 1);
    }
    const wo = 1 + Math.floor((wp - this.kernel.x) / this.strides[0]);
    const ho = 1 + Math.floor((hp - this.kernel.y) / this.strides[1]);
    this.biases = cpuZeroes2D([wo, ho]);
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
    this.input = reshape3D(inputTensor);
    if (this.padding > 0) {
      iterate1D(this.input.z, (z: number) => {
        iterate2D([this.input.x, this.input.y], (x: number, y: number) => {
          const idx = this.padded.index(this.padding + x, this.padding + y, z);
          this.padded.data[idx] = this.input.data[this.input.index(x, y, z)];
        });
      });
    } else {
      this.padded = this.input;
    }
    iterate1D(this.input.z, (z: number) => {
      iterate2D([this.output.x, this.output.y], (x: number, y: number) => {
        let sum = 0;
        iterate2D(this.kernel, (i: number, j: number) => {
          const w = x * this.strides[0] + i
          const h = y * this.strides[1] + j
          const k = this.padded.index(w, h, z);
          const l = j * this.kernel.x + i;
          sum += this.padded.data[k] * this.kernel.data[l];
        });
        const idx = this.output.index(x, y, z);
        sum += this.biases.data[y * this.output.x + x];
        this.output.data[idx] = this.activationFn.activate(sum);
      });
    });
    return this.output;
  }

  backPropagate(errorTensor: CPUTensor<Rank>, rate: number) {
    this.error = reshape3D(errorTensor);
    const cost = cpuZeroes3D(this.error.shape);
    for (const i in cost.data) {
      const activation = this.activationFn.prime(this.output.data[i]);
      cost.data[i] = this.error.data[i] * activation;
    }
    iterate1D(this.input.z, (z: number) => {
      iterate2D(this.kernel, (x: number, y: number) => {
        let sum = 0;
        iterate2D([cost.x, cost.y], (i: number, j: number) => {
          const w = x * this.strides[0] + i;
          const h = y * this.strides[1] + j;
          const k = this.padded.index(w, h, z);
          const l = cost.index(i, j, z);
          sum += this.padded.data[k] * cost.data[l];
        });
        const idx = y * this.kernel.x + x;
        this.kernel.data[idx] += this.activationFn.activate(sum) * rate;
      });
      for (let i = 0; i < this.biases.data.length; i++) {
        this.biases.data[i] += cost.data[i + z * cost.x * cost.y];
      }
    });
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
