import { WebGPUBackend } from "../../../deps.ts";
import {
  Activation,
  DenseLayerConfig,
  GPUTensor,
  LayerJSON,
  Rank,
  Shape,
  Shape1D,
} from "../../../core/types.ts";
import { to1D, to2D } from "../../../core/util.ts";
import { GPUActivationFn, setActivation, Sigmoid } from "../activation.ts";
import { CrossEntropy, GPUCostFunction } from "../cost.ts";
import { backPropagate } from "../kernels/backpropagate.ts";
import { feedForward } from "../kernels/feedforward.ts";
import { gpuZeroes2D, Tensor } from "../../../mod.ts";

/**
 * Regular Dense Layer
 */
export class DenseGPULayer {
  outputSize: Shape1D;
  activationFn: GPUActivationFn = new Sigmoid();
  costFunction: GPUCostFunction = new CrossEntropy();

  input!: GPUTensor<Rank.R2>;
  weights!: GPUTensor<Rank.R2>;
  biases!: GPUTensor<Rank.R2>;
  output!: GPUTensor<Rank.R2>;
  error!: GPUTensor<Rank.R2>;
  cost!: GPUTensor<Rank.R2>;

  #backend: WebGPUBackend;

  constructor(config: DenseLayerConfig, backend: WebGPUBackend) {
    this.outputSize = config.size;
    this.setActivation(config.activation || "linear");
    this.#backend = backend;
  }

  reset(batches: number) {
    if (batches != this.output.y) {
      this.output = gpuZeroes2D([this.outputSize[0], batches]);
      this.error = gpuZeroes2D([this.outputSize[0], batches]);
      this.cost = gpuZeroes2D([this.outputSize[0], batches]);
    }
  }

  initialize(inputShape: Shape[Rank]) {
    const shape = to2D(inputShape);
    const weights = new Float32Array(this.outputSize[0] * shape[0])
      .map(() => Math.random() * 2 - 1);
    const biases = new Float32Array(this.outputSize[0])
      .map(() => Math.random() * 2 - 1);
    if (!this.weights) {
      this.weights = gpuZeroes2D([this.outputSize[0], shape[0]]);
      this.biases = gpuZeroes2D([this.outputSize[0], 1]);
      this.output = gpuZeroes2D([this.outputSize[0], shape[1]]);
      this.error = gpuZeroes2D([this.outputSize[0], shape[1]]);
      this.cost = gpuZeroes2D([this.outputSize[0], shape[1]]);
    }
    this.weights.setData(weights);
    this.biases.setData(biases);
  }

  setActivation(activation: Activation) {
    this.activationFn = setActivation(activation);
  }

  async feedForward(input: GPUTensor<Rank>) {
    this.input = input.to2D();
    await feedForward(
      this.#backend,
      this.input,
      this.weights,
      this.biases,
      this.output,
      this.activationFn.activate(),
    );
    return this.output;
  }

  async backPropagate(
    error: GPUTensor<Rank>,
    prev: GPUTensor<Rank>,
    rate: number,
    last: number,
    costFn: GPUCostFunction = this.costFunction,
  ) {
    await backPropagate(
      this.#backend,
      this.input,
      this.weights,
      this.biases,
      this.output,
      this.cost,
      error.to2D(),
      this.error,
      prev.to2D(),
      rate,
      last,
      this.activationFn.prime(),
      costFn.prime(),
    );
    return this.output;
  }

  async toJSON(): Promise<LayerJSON> {
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
    { outputSize, type, activationFn, weights, biases }: LayerJSON,
    backend: WebGPUBackend,
  ) {
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
    const layer = new DenseGPULayer({
      size: to1D(outputSize!),
      activation: (activationFn as Activation) || "sigmoid",
    }, backend);
    layer.weights = Tensor.fromJSON(weights);
    layer.biases = Tensor.fromJSON(biases);
    return layer;
  }
}
