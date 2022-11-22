import { setInit, Xavier } from "../../../core/init.ts";
import {
  cpuZeroes2D,
  reshape2D,
  Tensor,
  toShape2D,
} from "../../../core/tensor.ts";
import {
  CPUTensor,
  DenseLayerConfig,
  InitFn,
  LayerJSON,
  Rank,
  Shape,
  Shape1D,
  Shape2D,
} from "../../../core/types.ts";
import { CPUMatrix } from "../kernels/matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Regular Dense Layer
 */
export class DenseCPULayer {
  type = "dense";
  outputSize: Shape1D;
  init: InitFn = new Xavier();

  input!: CPUTensor<Rank.R2>;
  weights!: CPUTensor<Rank.R2>;
  biases!: CPUTensor<Rank.R2>;
  output!: CPUTensor<Rank.R2>;

  constructor(config: DenseLayerConfig) {
    this.outputSize = config.size;
    this.init = setInit(config.init || "uniform");
  }

  reset(batches: number) {
    this.output = cpuZeroes2D([this.outputSize[0], batches]);
  }

  initialize(inputShape: Shape[Rank]) {
    const shape = toShape2D(inputShape);
    const weights = [this.outputSize[0], shape[0]] as Shape2D;
    this.weights = this.init.init([shape[0]], weights, this.outputSize);
    this.biases = cpuZeroes2D([this.outputSize[0], 1]);
    this.reset(shape[1]);
  }

  feedForward(inputTensor: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = reshape2D(inputTensor);
    const product = CPUMatrix.dot(this.input, this.weights);
    for (let i = 0, j = 0; i < product.data.length; i++, j++) {
      if (j >= this.biases.x) j = 0;
      this.output.data[i] = product.data[i] + this.biases.data[j];
    }
    return this.output;
  }

  backPropagate(
    errorTensor: CPUTensor<Rank>,
    rate: number,
  ) {
    const dError = reshape2D(errorTensor);
    const dInput = CPUMatrix.dot(dError, CPUMatrix.transpose(this.weights));
    const dWeights = CPUMatrix.dot(CPUMatrix.transpose(this.input), dError);
    for (const i in dWeights.data) {
      this.weights.data[i] -= dWeights.data[i] * rate;
    }
    for (let i = 0, j = 0; i < dError.data.length; i++, j++) {
      if (j >= this.biases.x) j = 0;
      this.biases.data[j] -= dError.data[i] * rate;
    }
    return dInput;
  }

  async toJSON() {
    return {
      outputSize: this.outputSize,
      type: this.type,
      weights: await this.weights.toJSON(),
      biases: await this.biases.toJSON(),
    };
  }

  static fromJSON(
    { outputSize, weights, biases }: LayerJSON,
  ): DenseCPULayer {
    if (!weights || !biases) {
      throw new Error("Layer imported must be initialized");
    }
    const layer = new DenseCPULayer({ size: outputSize as Shape1D });
    layer.weights = Tensor.fromJSON(weights);
    layer.biases = Tensor.fromJSON(biases);
    return layer;
  }
}
