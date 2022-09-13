import { DataType } from "../../../deps.ts";
import { Activation, ConvLayerConfig, Size2D } from "../../types.ts";
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
  padding = 0;
  stride = 1;
  activationFn: CPUActivationFn = new Sigmoid();

  input!: CPUMatrix;
  weights!: CPUMatrix;
  biases!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: ConvLayerConfig) {
    this.outputSize = config.size as Size2D;
    this.setActivation(config.activation);
  }

  reset(type: DataType, inputSize: Size2D, kernelSize: Size2D) {
    const w = 1 + (inputSize.x + 2 * this.padding - kernelSize.x) / this.stride;
    const h = 1 + (inputSize.y + 2 * this.padding - kernelSize.y) / this.stride;
    this.output = CPUMatrix.with(w, h, type);
  }

  initialize(type: DataType, inputSize: Size2D, kernelSize: Size2D) {
    this.reset(type, inputSize, kernelSize);
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
