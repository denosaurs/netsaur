import {
  Activation,
  DenseLayerConfig,
  LayerJSON,
  MatrixJSON,
  Size,
} from "../../../core/types.ts";
import { ActivationError, to1D } from "../../../core/util.ts";
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
import { CPUCostFunction, CrossEntropy } from "../cost.ts";
import { CPUMatrix } from "../matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Base class for all layers.
 */
export class DenseCPULayer {
  outputSize: number;
  activationFn: CPUActivationFn = new Sigmoid();
  costFunction: CPUCostFunction = new CrossEntropy();

  input!: CPUMatrix;
  weights!: CPUMatrix;
  biases!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: DenseLayerConfig) {
    this.outputSize = to1D(config.size);
    this.setActivation(config.activation);
  }

  reset(batches: number) {
    this.output = CPUMatrix.with(this.outputSize, batches);
  }

  initialize(inputSize: Size, batches: number) {
    this.weights = CPUMatrix.with(this.outputSize, to1D(inputSize));
    this.weights.data = this.weights.data.map(() => Math.random() * 2 - 1);
    this.biases = CPUMatrix.with(this.outputSize, 1);
    this.biases.data = this.biases.data.map(() => Math.random() * 2 - 1);
    this.reset(batches);
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

  backPropagate(error: CPUMatrix, rate: number, _costFn: CPUCostFunction = this.costFunction,) {
    const cost = CPUMatrix.with(error.x, error.y);
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

  toJSON(): LayerJSON {
    return {
      outputSize: this.outputSize,
      activationFn: this.activationFn.name,
      costFn: this.costFunction.name,
      type: "dense",
      input: this.input.toJSON(),
      weights: this.weights.toJSON(),
      biases: this.biases.toJSON(),
      output: this.output.toJSON(),
    };
  }

  static fromJSON(
    { outputSize, activationFn, input, weights, biases, output }: LayerJSON,
  ): DenseCPULayer {
    const layer = new DenseCPULayer({
      size: outputSize,
      activation: (activationFn as Activation) || "sigmoid",
    });
    layer.input = new CPUMatrix(input.data, input.x, input.y);
    layer.weights = new CPUMatrix(
      (weights as MatrixJSON).data,
      (weights as MatrixJSON).x,
      (weights as MatrixJSON).y,
    );
    layer.biases = new CPUMatrix(
      (biases as MatrixJSON).data,
      (biases as MatrixJSON).x,
      (biases as MatrixJSON).y,
    );
    layer.output = new CPUMatrix(output.data, output.x, output.y);
    return layer;
  }
}
