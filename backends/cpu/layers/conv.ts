import {
  Activation,
  ConvLayerConfig,
  CPUTensor,
  LayerJSON,
  Rank,
  Shape,
  Shape2D,
  Shape3D,
  Shape4D,
} from "../../../core/types.ts";
import {
  iterate1D,
  iterate2D,
  iterate3D,
  iterate4D,
  Random,
} from "../../../core/util.ts";
import {
  cpuZeroes1D,
  cpuZeroes4D,
  reshape4D,
  Tensor,
  toShape4D,
} from "../../../mod.ts";
import { CPUActivationFn, setActivation, Sigmoid } from "../activation.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Convolutional 2D layer.
 */
export class ConvCPULayer {
  config: ConvLayerConfig;
  outputSize!: Shape3D;
  paddedSize!: Shape3D;
  padding: number;
  strides: Shape2D;
  activationFn: CPUActivationFn = new Sigmoid();

  input!: CPUTensor<Rank.R4>;
  kernel!: CPUTensor<Rank.R4>;
  padded!: CPUTensor<Rank.R4>;
  biases!: CPUTensor<Rank.R1>;
  sum!: CPUTensor<Rank.R4>;
  error!: CPUTensor<Rank.R4>;
  output!: CPUTensor<Rank.R4>;

  constructor(config: ConvLayerConfig) {
    this.config = config;
    if (config.kernel) {
      this.kernel = new Tensor(config.kernel, config.kernelSize);
    } else {
      this.kernel = cpuZeroes4D(config.kernelSize);
    }
    this.padding = config.padding || 0;
    this.strides = config.strides || [1, 1];
    this.setActivation(config.activation ?? "linear");
  }

  reset(batches: number) {
    const [wp, hp, c] = this.paddedSize;
    const [wo, ho, f] = this.outputSize;
    this.sum = cpuZeroes4D([wo, ho, f, batches]);
    this.output = cpuZeroes4D([wo, ho, f, batches]);
    if (this.padding > 0) {
      const data = new Float32Array(wp * hp * batches).fill(0);
      this.padded = new Tensor(data, [wp, hp, c, batches]);
    }
  }

  initialize(inputSize: Shape[Rank]) {
    const size = toShape4D(inputSize);
    const wp = size[0] + 2 * this.padding;
    const hp = size[1] + 2 * this.padding;
    this.paddedSize = [wp, hp, size[2]];
    if (!this.config.kernel) {
      const sd = Math.sqrt(1 / (size[2] * this.kernel.x * this.kernel.y));
      this.kernel.data = this.kernel.data.map(() => Random.gaussian(0, sd));
    }
    const wo = 1 + Math.floor((wp - this.kernel.x) / this.strides[0]);
    const ho = 1 + Math.floor((hp - this.kernel.y) / this.strides[1]);
    this.biases = cpuZeroes1D([this.kernel.shape[3]]);
    if (!this.config.unbiased) {
      this.biases.data = this.biases.data.map(() => 0);
    }
    this.outputSize = [wo, ho, this.kernel.shape[3]];
    this.reset(size[3]);
  }

  setActivation(activation: Activation) {
    this.activationFn = setActivation(activation);
  }

  feedForward(inputTensor: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = reshape4D(inputTensor);
    if (this.padding > 0) {
      iterate4D(this.input, (x, y, z, w) => {
        const idx = this.padded.index(this.padding + x, this.padding + y, z, w);
        this.padded.data[idx] = this.input.data[this.input.index(x, y, z, w)];
      });
    } else {
      this.padded = this.input;
    }
    iterate4D(this.output, (x, y, z, w) => {
      let sum = this.biases.data[0];
      iterate3D(this.kernel, (i, j, k) => {
        const W = x * this.strides[0] + i;
        const H = y * this.strides[1] + j;
        const P = this.padded.index(W, H, k, w);
        const K = this.kernel.index(i, j, k, z);
        sum += this.padded.data[P] * this.kernel.data[K];
      });
      const idx = this.output.index(x, y, z, w);
      this.sum.data[idx] = sum;
      this.output.data[idx] = this.activationFn.activate(sum);
    });
    return this.output;
  }

  backPropagate(errorTensor: CPUTensor<Rank>, rate: number) {
    this.error = reshape4D(errorTensor);
    const cost = cpuZeroes4D(this.error.shape);
    for (const i in cost.data) {
      const activation = this.activationFn.prime(this.sum.data[i]);
      cost.data[i] = this.error.data[i] * activation;
    }
    const dInput = cpuZeroes4D(this.padded.shape);
    iterate4D(dInput, (x, y, z, w) => {
      let sum = 0;
      iterate3D(cost, (i, j, k) => {
        const W = x * this.strides[0] + i;
        const H = y * this.strides[1] + j;
        if (W >= 0 && H >= 0 && W < this.kernel.x && H < this.kernel.x) {
          const K = this.kernel.index(H, W, z, k);
          const C = cost.index(i, j, k, w);
          sum += this.kernel.data[K] * cost.data[C];
        }
      });
      const idx = dInput.index(x, y, z, w);
      dInput.data[idx] = sum;
    });
    iterate1D(cost.w, (b) => {
      iterate4D(this.kernel, (x, y, z, w) => {
        let sum = 0;
        iterate2D([cost.x, cost.y], (i, j) => {
          const W = x * this.strides[0] + i;
          const H = y * this.strides[1] + j;
          const P = this.padded.index(W, H, z, b);
          const C = cost.index(i, j, w, b);
          sum += this.padded.data[P] * cost.data[C];
        });
        const idx = this.kernel.index(x, y, z, w);
        this.kernel.data[idx] -= sum * rate;
      });
    });
    iterate1D(cost.z, (z) => {
      let sum = 0;
      iterate3D([cost.x, cost.y, cost.w], (x, y, w) => {
        sum += cost.data[cost.index(x, y, z, w)];
      });
      this.biases.data[z] -= sum * rate;
    });
    return dInput;
  }

  async toJSON() {
    return {
      outputSize: this.outputSize,
      type: "conv",
      kernel: await this.kernel.toJSON(),
      biases: await this.biases.toJSON(),
      strides: this.strides,
      paddedSize: this.paddedSize,
      padding: this.padding,
      activationFn: this.activationFn
    };
  }

  static fromJSON(
    { outputSize, kernel, type, biases, strides, padding, paddedSize, activationFn }:
      LayerJSON,
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
      kernelSize: kernel.shape as Shape4D,
      padding,
      strides: strides as Shape2D,
      activation: activationFn as Activation
    });
    layer.paddedSize = paddedSize as Shape3D;
    layer.outputSize = outputSize as Shape3D;
    layer.kernel = Tensor.fromJSON(kernel);
    layer.biases = Tensor.fromJSON(biases);
    return layer;
  }
}
