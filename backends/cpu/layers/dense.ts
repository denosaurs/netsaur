import { Tensor } from "../../../core/tensor.ts";
import {
  Activation,
  BackendType,
  CPUTensor,
  DenseLayerConfig,
  LayerJSON,
  MatrixJSON,
  Rank,
  Shape,
  Shape1D,
} from "../../../core/types.ts";
import { ActivationError, toShape } from "../../../core/util.ts";
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
import { CPUMatrix } from "../kernels/matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Regular Dense Layer
 */
export class DenseCPULayer {
  outputSize: Shape1D;
  activationFn: CPUActivationFn = new Sigmoid();
  costFunction: CPUCostFunction = new CrossEntropy();

  input!: CPUTensor<Rank.R2>;
  weights!: CPUTensor<Rank.R2>;
  biases!: CPUTensor<Rank.R2>;
  output!: CPUTensor<Rank.R2>;
  error!: CPUTensor<Rank.R2>;

  constructor(config: DenseLayerConfig) {
    this.outputSize = config.size;
    this.setActivation(config.activation || "linear");
  }

  reset(batches: number) {
    this.output = Tensor.zeroes([this.outputSize[0], batches]);
  }

  initialize(inputShape: Shape[Rank]) {
    const shape = toShape(inputShape, Rank.R2);
    this.weights = Tensor.zeroes([this.outputSize[0], shape[0]]);
    this.weights.data = this.weights.data.map(() => Math.random() * 2 - 1);
    this.biases = Tensor.zeroes([this.outputSize[0], 1]);
    this.biases.data = this.biases.data.map(() => Math.random() * 2 - 1);
    this.reset(shape[1]);
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

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input.to2D();
    const product = CPUMatrix.dot(this.input, this.weights);
    for (let i = 0, j = 0; i < product.data.length; i++, j++) {
      if (j >= this.biases.x) j = 0;
      const sum = product.data[i] + this.biases.data[j];
      this.output.data[i] = this.activationFn.activate(sum);
    }
    return this.output;
  }

  backPropagate(
    error: CPUTensor<Rank>,
    rate: number,
  ) {
    const cost = Tensor.zeroes<Rank.R2, BackendType.CPU>([error.x, error.y]);
    for (let i = 0; i < cost.data.length; i++) {
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
    this.error = error.to2D();
  }

  getError(): CPUTensor<Rank> {
    return CPUMatrix.dot(this.error, CPUMatrix.transpose(this.weights));
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
    { outputSize, activationFn, type, input, weights, biases, output }:
      LayerJSON,
  ): DenseCPULayer {
    if (type !== "dense") {
      throw new Error(
        "Cannot cannot create a Dense layer from a" +
          type.charAt(0).toUpperCase() + type.slice(1) +
          "Layer",
      );
    }
    if (weights === undefined || biases === undefined) {
      throw new Error("Layer imported must be initialized");
    }
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
