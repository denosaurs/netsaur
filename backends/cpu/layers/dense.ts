import {
  cpuZeroes2D,
  reshape2D,
  Tensor,
  toShape2D,
} from "../../../core/tensor.ts";
import {
  Activation,
  CPUTensor,
  DenseLayerConfig,
  LayerJSON,
  Rank,
  Shape,
  Shape1D,
} from "../../../core/types.ts";
import { Random } from "../../../core/util.ts";
import { CPUActivationFn, setActivation, Sigmoid } from "../activation.ts";
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
  sum!: CPUTensor<Rank.R2>;
  output!: CPUTensor<Rank.R2>;
  error!: CPUTensor<Rank.R2>;

  constructor(config: DenseLayerConfig) {
    this.outputSize = config.size;
    this.setActivation(config.activation || "linear");
  }

  reset(batches: number) {
    this.sum = cpuZeroes2D([this.outputSize[0], batches]);
    this.output = cpuZeroes2D([this.outputSize[0], batches]);
  }

  initialize(inputShape: Shape[Rank]) {
    const shape = toShape2D(inputShape);
    this.weights = cpuZeroes2D([this.outputSize[0], shape[0]]);
    this.weights.data = this.weights.data.map(() => Random.gaussian(0) * 0.01);
    this.biases = cpuZeroes2D([this.outputSize[0], 1]);
    this.biases.data = this.biases.data.map(() => 0);
    this.reset(shape[1]);
  }

  setActivation(activation: Activation) {
    this.activationFn = setActivation(activation);
  }

  feedForward(inputTensor: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = reshape2D(inputTensor);
    const product = CPUMatrix.dot(this.input, this.weights);
    for (let i = 0, j = 0; i < product.data.length; i++, j++) {
      if (j >= this.biases.x) j = 0;
      this.sum.data[i] = product.data[i] + this.biases.data[j];
      this.output.data[i] = this.activationFn.activate(this.sum.data[i]);
    }
    return this.output;
  }

  backPropagate(
    errorTensor: CPUTensor<Rank>,
    rate: number,
  ) {
    this.error = reshape2D(errorTensor);
    const cost = cpuZeroes2D([this.error.x, this.error.y]);
    for (let i = 0; i < cost.data.length; i++) {
      const activation = this.activationFn.prime(this.sum.data[i]);
      cost.data[i] = this.error.data[i] * activation;
    }
    const dInput = CPUMatrix.dot(this.error, CPUMatrix.transpose(this.weights))
    const weightsDelta = CPUMatrix.dot(CPUMatrix.transpose(this.input), cost);
    for (const i in weightsDelta.data) {
      this.weights.data[i] -= weightsDelta.data[i] * rate;
    }
    for (let i = 0, j = 0; i < this.error.data.length; i++, j++) {
      if (j >= this.biases.x) j = 0;
      this.biases.data[j] -= cost.data[i] * rate;
    }
    return dInput
  }

  async toJSON() {
    return {
      outputSize: this.outputSize,
      activationFn: this.activationFn.name,
      costFn: this.costFunction.name,
      type: "dense",
      weights: await this.weights.toJSON(),
      biases: await this.biases.toJSON(),
    };
  }

  static fromJSON(
    { outputSize, activationFn, type, weights, biases }: LayerJSON,
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
      size: outputSize as Shape1D,
      activation: (activationFn as Activation) || "sigmoid",
    });
    layer.weights = Tensor.fromJSON(weights);
    layer.biases = Tensor.fromJSON(biases);
    return layer;
  }
}
