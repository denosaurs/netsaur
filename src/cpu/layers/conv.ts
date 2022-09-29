import { Activation, ConvLayerConfig, LayerJSON, Size, Size2D } from "../../types.ts";
import { ActivationError, iterate2D } from "../../util.ts";
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
 * Convolutional layer.
 */
export class ConvCPULayer {
  outputSize: Size2D;
  padding: number;
  stride: number;
  activationFn: CPUActivationFn = new Sigmoid();

  input!: CPUMatrix;
  kernel!: CPUMatrix;
  padded!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: ConvLayerConfig) {
    this.outputSize = config.size as Size2D;
    this.kernel = new CPUMatrix(
      config.kernel,
      config.kernelSize.x,
      config.kernelSize.y,
    );
    this.padding = config.padding || 0;
    this.stride = config.stride || 1;
    this.setActivation(config.activation);
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Size, _batches: number) {
    const size = inputSize as Size2D;
    const wp = size.x + 2 * this.padding;
    const hp = size.y + 2 * this.padding;
    if (this.padding > 0) this.padded = CPUMatrix.with(wp, hp);
    const wo = 1 + Math.floor((wp - this.kernel.x) / this.stride);
    const ho = 1 + Math.floor((hp - this.kernel.y) / this.stride);
    this.output = CPUMatrix.with(wo, ho);
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
        const k = this.padded.x * (j * this.stride + y) + (i * this.stride + x);
        const l = y * this.kernel.x + x;
        sum += this.padded.data[k] * this.kernel.data[l];
      });
      this.output.data[j * this.output.x + i] = sum;
    });
    return this.output;
  }

  backPropagate(_error: CPUMatrix, _rate: number) {
  }

  toJSON(): LayerJSON {
    return {
      outputSize: this.outputSize,
      activation: this.activationFn,
      type: "conv"
    };
  }
}
