import {
  Activation,
  ConvLayerConfig,
  LayerJSON,
  MatrixJSON,
  Size,
  Size2D,
} from "../../../core/types.ts";
import { ActivationError, iterate2D, to2D } from "../../../core/util.ts";
import {
  CPUActivationFn,
  Elu,
  LeakyRelu,
  Linear,
  Relu,
  Relu6,
  Selu,
  Sigmoid,
  Tanh,
} from "../activation.ts";
import { CPUMatrix } from "../matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Convolutional 2D layer.
 */
export class ConvCPULayer {
  config: ConvLayerConfig;
  outputSize!: Size2D;
  padding: number;
  strides: Size2D;
  activationFn: CPUActivationFn = new Sigmoid();

  input!: CPUMatrix;
  kernel!: CPUMatrix;
  padded!: CPUMatrix;
  biases!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: ConvLayerConfig) {
    this.config = config;
    const [ y, x ] = config.kernelSize;
    if (config.kernel) {
      this.kernel = new CPUMatrix(config.kernel, x, y);
    } else {
      this.kernel = CPUMatrix.with(x, y);
    }
    this.padding = config.padding || 0;
    this.strides = to2D(config.strides);
    this.setActivation(config.activation ?? "linear");
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Size, _batches: number) {
    const wp = (inputSize as Size2D)[1] /**x */ + 2 * this.padding;
    const hp = (inputSize as Size2D)[0] /**y */ + 2 * this.padding;
    if (this.padding > 0) {
      this.padded = CPUMatrix.with(wp, hp);
      this.padded.fill(255);
    }
    if (!this.config.kernel) {
      this.kernel.data = this.kernel.data.map(() => Math.random() * 2 - 1);
    }
    const wo = 1 + Math.floor((wp - this.kernel.x) / this.strides[1]);
    const ho = 1 + Math.floor((hp - this.kernel.y) / this.strides[0]);
    this.biases = CPUMatrix.with(wo, ho);
    if (!this.config.unbiased) {
      this.biases.data = this.biases.data.map(() => Math.random() * 2 - 1);
    }
    this.output = CPUMatrix.with(wo, ho);
    this.outputSize = [ho, wo];
  }

  setActivation(activation: Activation) {
    switch (activation) {
      case "sigmoid":
        this.activationFn = new Sigmoid();
        break;
      case "leakyrelu":
        this.activationFn = new LeakyRelu();
        break;
      case "tanh":
        this.activationFn = new Tanh();
        break;
      case "relu":
        this.activationFn = new Relu();
        break;
      case "relu6":
        this.activationFn = new Relu6();
        break;
      case "elu":
        this.activationFn = new Elu();
        break;
      case "selu":
        this.activationFn = new Selu();
        break;
      case "linear":
        this.activationFn = new Linear();
        break;
      default:
        throw new ActivationError(activation);
    }
  }

  feedForward(input: CPUMatrix): CPUMatrix {
    if (this.padding > 0) {
      iterate2D(input, (i: number, j: number) => {
        const idx = this.padded.x * (this.padding + j) + this.padding + i;
        this.padded.data[idx] = input.data[j * input.x + i];
      });
    } else {
      this.padded = input;
    }
    iterate2D(this.output, (i: number, j: number) => {
      let sum = 0;
      iterate2D(this.kernel, (x: number, y: number) => {
        const k = this.padded.x * (j * this.strides[0] + y) +
          (i * this.strides[1] + x);
        const l = y * this.kernel.x + x;
        sum += this.padded.data[k] * this.kernel.data[l];
      });
      sum += this.biases.data[j * this.output.x + i];
      this.output.data[j * this.output.x + i] = this.activationFn.activate(sum);
    });
    return this.output;
  }

  backPropagate(error: CPUMatrix, rate: number) {
    const cost = CPUMatrix.with(error.x, error.y);
    for (const i in cost.data) {
      const activation = this.activationFn.prime(this.output.data[i]);
      cost.data[i] = error.data[i] * activation;
    }
    iterate2D(this.kernel, (i: number, j: number) => {
      let sum = 0;
      iterate2D(cost, (x: number, y: number) => {
        const k = this.padded.x * (j * this.strides[0] + y) +
          (i * this.strides[1] + x);
        const l = y * cost.x + x;
        sum += this.padded.data[k] * cost.data[l];
      });
      const idx = j * this.kernel.x + i;
      this.kernel.data[idx] += this.activationFn.activate(sum) * rate;
    });
    for (let i = 0; i < cost.data.length; i++) {
      this.biases.data[i] += cost.data[i]
    }
  }

  toJSON(): LayerJSON {
    return {
      outputSize: this.outputSize,
      type: "conv",
      input: this.input.toJSON(),
      kernel: this.kernel.toJSON(),
      padded: this.padded.toJSON(),
      output: this.output.toJSON(),
      strides: this.strides,
      padding: this.padding,
    };
  }

  static fromJSON(
    { outputSize, input, kernel, type, padded, output, strides, padding }:
      LayerJSON,
  ): ConvCPULayer {
    if (type !== "conv") {
      throw new Error(
        "Cannot cannot create a Convolutional layer from a" +
          type.charAt(0).toUpperCase() + type.slice(1) +
          "Layer",
      );
    }
    if (padded === undefined || kernel === undefined) {
      throw new Error("Layer imported must be initialized");
    }
    const layer = new ConvCPULayer({
      kernel: (kernel as MatrixJSON).data,
      kernelSize: [ (kernel as MatrixJSON).y, (kernel as MatrixJSON).x ],
      padding,
      strides,
    });
    layer.input = new CPUMatrix(input.data, input.x, input.y);
    layer.outputSize = outputSize as Size2D;
    layer.padded = new CPUMatrix(
      (padded as MatrixJSON).data,
      (padded as MatrixJSON).x,
      (padded as MatrixJSON).y,
    );
    layer.output = new CPUMatrix(output.data, output.x, output.y);
    return layer;
  }
}
