import { DataType, WebGPUBackend } from "../../../deps.ts";
import { Activation, LayerConfig } from "../../types.ts";
import { ActivationError, fromType } from "../../util.ts";
import {
  Elu,
  GPUActivationFn,
  LeakyRelu,
  Linear,
  Relu,
  Relu6,
  Selu,
  Sigmoid,
  Tanh,
} from "../activation.ts";
import { CrossEntropy, GPUCostFunction } from "../cost.ts";
import { backPropagate } from "../kernels/backpropagate.ts";
import { feedForward } from "../kernels/feedforward.ts";
import { GPUMatrix } from "../matrix.ts";

interface GPULayerConfig extends LayerConfig {
  size: number;
  activation: Activation;
}
/**
 * Base class for all layers.
 */
export class BaseGPULayer {
  outputSize: number;
  activationFn: GPUActivationFn = new Sigmoid();
  costFunction: GPUCostFunction = new CrossEntropy();

  inputs!: GPUMatrix;
  weights!: GPUMatrix;
  biases!: GPUMatrix;
  output!: GPUMatrix;
  error!: GPUMatrix;
  cost!: GPUMatrix;

  #backend: WebGPUBackend;

  constructor(config: GPULayerConfig, backend: WebGPUBackend) {
    this.outputSize = config.size;
    this.setActivation(config.activation);
    this.#backend = backend;
  }

  async reset(type: DataType, batches: number) {
    const b = this.#backend;
    if (batches != this.output.y) {
      this.output = await GPUMatrix.with(b, this.outputSize, batches, type);
      this.error = await GPUMatrix.with(b, this.outputSize, batches, type);
      this.cost = await GPUMatrix.with(b, this.outputSize, batches, type);
    }
  }

  async initialize(type: DataType, inputSize: number, batches: number) {
    const b = this.#backend;
    const weights = new (fromType(type))(this.outputSize * inputSize)
      .map(() => Math.random() * 2 - 1);
    const biases = new (fromType(type))(this.outputSize)
      .map(() => Math.random() * 2 - 1);
    if (!this.weights) {
      this.weights = await GPUMatrix.with(b, this.outputSize, inputSize, type);
      this.biases = await GPUMatrix.with(b, this.outputSize, 1, type);
      this.output = await GPUMatrix.with(b, this.outputSize, batches, type);
      this.error = await GPUMatrix.with(b, this.outputSize, batches, type);
      this.cost = await GPUMatrix.with(b, this.outputSize, batches, type);
    }
    b.device.queue.writeBuffer(this.weights.data.buffer, 0, weights);
    b.device.queue.writeBuffer(this.biases.data.buffer, 0, biases);
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

  async feedForward(input: GPUMatrix): Promise<GPUMatrix> {
    this.inputs = input;
    await feedForward(
      this.#backend,
      this.inputs,
      this.weights,
      this.biases,
      this.output,
      this.activationFn.activate(input.type),
    );
    return this.output;
  }

  async backPropagate(
    error: GPUMatrix,
    prev: GPUMatrix,
    rate: number,
    last: number,
  ): Promise<GPUMatrix> {
    await backPropagate(
      this.#backend,
      this.inputs,
      this.weights,
      this.biases,
      this.output,
      this.cost,
      error,
      this.error,
      prev,
      rate,
      last,
      this.activationFn.prime(error.type),
      this.costFunction.prime(error.type),
    );
    return this.output;
  }

  toJSON() {
    return {
      outputSize: this.outputSize,
      activation: this.activationFn,
    };
  }
}
