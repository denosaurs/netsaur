import { DataType } from "../../../deps.ts";
import { Activation, LayerConfig } from "../../types.ts";
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
 * Base class for all layers.
 */
export class BaseCPULayer {
  outputSize: number;
  activationFn: CPUActivationFn = new Sigmoid();

  input!: CPUMatrix;
  weights!: CPUMatrix;
  biases!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: LayerConfig) {
    this.outputSize = config.size;
    this.setActivation(config.activation);
  }

  reset(type: DataType, batches: number) {
    this.output = CPUMatrix.with(this.outputSize, batches, type);
  }

  initialize(type: DataType, inputSize: number, batches: number) {
    this.weights = CPUMatrix.with(this.outputSize, inputSize, type);
    this.weights.data = this.weights.data.map(() => Math.random() * 2 - 1);
    this.biases = CPUMatrix.with(this.outputSize, 1, type);
    this.biases.data = this.biases.data.map(() => Math.random() * 2 - 1);
    this.reset(type, batches);
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
    this.input = input;
    const product = CPUMatrix.dot(input, this.weights);
    for (let i = 0, j = 0; i < product.data.length; i++, j++) {
      if (j >= this.biases.x) j = 0;
      const sum = product.data[i] + this.biases.data[j];
      this.output.data[i] = this.activationFn.activate(sum);
    }
    return this.output;
  }

  backPropagate(error: CPUMatrix, rate: number) {
    const cost = CPUMatrix.with(error.x, error.y, error.type);
    for (const i in cost.data) {
      const activation = this.activationFn.prime(this.output.data[i]);
      cost.data[i] = error.data[i] * activation;
    }
    const weightsDelta = CPUMatrix.dot(CPUMatrix.transpose(this.input), cost);
    for (const i in weightsDelta.data) {
      this.weights.data[i] += weightsDelta.data[i] * rate;
    }
    for (let i = 0, j = 0; i < error.data.length; i++, j++) {
      if (j >= this.biases.x) j = 0;
      this.biases.data[j] += cost.data[i] * rate;
    }
  }

  toJSON() {
    return {
      outputSize: this.outputSize,
      activation: this.activationFn,
    };
  }
}
