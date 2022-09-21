import { Activation, ConvLayerConfig, Size, Size2D } from "../../types.ts";
import { ActivationError } from "../../util.ts";
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
  test!: CPUMatrix;
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

  reset(batches: number) {
  }

  initialize(inputSize: Size, batches: number) {
    const size = inputSize as Size2D;
    const w = (size.x + 2 * this.padding) / this.stride;
    const h = (size.y + 2 * this.padding) / this.stride;
    this.test = CPUMatrix.with(w, h);
    const w2 = 1 + (size.x + 2 * this.padding - this.kernel.x) / this.stride;
    const h2 = 1 + (size.y + 2 * this.padding - this.kernel.y) / this.stride;
    this.output = CPUMatrix.with(w2, h2);
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

  feedForward(input: CPUMatrix) {
    for (let i = 0; i < input.x; i++) {
      for (let j = 0; j < input.y; j++) {
        const idx = this.test.x * (this.padding + j) + this.padding + i;
        this.test.data[idx] = input.data[j * input.x + i];
      }
    }
    for (let i = 0; i < this.output.x; i++) {
      for (let j = 0; j < this.output.y; j++) {
        let sum = 0
        for (let x = 0; x < this.kernel.x; x++) {
          for (let y = 0; y < this.kernel.y; y++) {
            const k = this.test.x * (j + y) + (i + x)
            const l = y * this.kernel.x + x
            sum += this.test.data[k] * this.kernel.data[l]
          }
        }
        this.output.data[j * this.output.x + i] = sum;
      }
    }
    return this.output;
  }

  backPropagate(error: CPUMatrix, rate: number) {
  }

  toJSON() {
    return {
      outputSize: this.outputSize,
      activation: this.activationFn,
    };
  }
}
