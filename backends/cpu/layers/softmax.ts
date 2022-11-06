import { Tensor } from "../../../core/tensor.ts";
import {
  Activation,
  CPUTensor,
  LayerJSON,
  Rank,
  Shape,
} from "../../../core/types.ts";
import { iterate1D } from "../../../core/util.ts";
// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Softmax Layer
 */
export class SoftmaxCPULayer {
  outputSize!: Shape[Rank];
  output!: CPUTensor<Rank>;

  constructor() {
  }

  reset(batches: number) {
    let shape = batches;
    iterate1D(this.outputSize.length, (i) => shape *= this.outputSize[i]);
    const outputShape = [...this.outputSize, batches] as Shape[Rank];
    this.output = new Tensor(new Float32Array(shape), outputShape);
  }

  initialize(shape: Shape[Rank]) {
    this.outputSize = shape.slice(0, -1) as Shape[Rank];
    this.reset(shape[shape.length - 1]);
  }

  setActivation(_activation: Activation) {
  }

  feedForward(inputTensor: CPUTensor<Rank>): CPUTensor<Rank> {
    let sum = 0;
    iterate1D(inputTensor.data.length, (i) => {
      this.output.data[i] = Math.exp(inputTensor.data[i]);
      sum += this.output.data[i];
    });
    iterate1D(inputTensor.data.length, (i) => {
      this.output.data[i] /= sum;
    });
    return this.output;
  }

  backPropagate(
    errorTensor: CPUTensor<Rank>,
    _rate: number,
  ) {
    return errorTensor;
  }

  // deno-lint-ignore require-await
  async toJSON() {
    return {
      outputSize: this.outputSize,
      type: "softmax",
    };
  }

  static fromJSON(
    { type, outputSize }: LayerJSON,
  ): SoftmaxCPULayer {
    if (type !== "softmax") {
      throw new Error(
        "Cannot cannot create a Dense layer from a" +
          type.charAt(0).toUpperCase() + type.slice(1) +
          "Layer",
      );
    }
    const layer = new SoftmaxCPULayer();
    layer.outputSize = outputSize
    return layer;
  }
}
