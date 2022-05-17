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
import { feedForward } from "../kernels/feedforward.ts";
// import { backPropagate } from "../kernels/backPropagate.ts";
import { GPUMatrix } from "../matrix.ts";

interface GPULayerConfig extends LayerConfig {
  size: number;
  activation: Activation;
}
/**
 * Base class for all layers.
 */
export class BaseGPULayer<T extends DataType = DataType> {
  outputSize: number;
  activationFn: GPUActivationFn = new Sigmoid();
  costFunction: GPUCostFunction = new CrossEntropy();

  inputs!: GPUMatrix;
  weights!: GPUMatrix;
  product!: GPUMatrix;
  output!: GPUMatrix;
  weightsDelta!: GPUMatrix;
  error!: GPUMatrix;

  #backend: WebGPUBackend;

  constructor(config: GPULayerConfig, backend: WebGPUBackend) {
    this.outputSize = config.size;
    this.setActivation(config.activation);
    this.#backend = backend;
  }

  async reset(type: DataType, batches: number) {
    this.output = await GPUMatrix.with(
      this.#backend,
      this.outputSize,
      batches,
      type,
    );
    this.product = await GPUMatrix.with(
      this.#backend,
      this.outputSize,
      batches,
      type,
    );
  }

  async initialize(type: DataType, inputSize: number, batches: number) {
    const data = new (fromType(type))(this.outputSize * inputSize).fill(1);

    this.weights = await GPUMatrix.from(
      this.#backend,
      data,
      this.outputSize,
      inputSize,
      type,
    );
    this.output = await GPUMatrix.with(
      this.#backend,
      this.outputSize,
      batches,
      type,
    );
    this.product = await GPUMatrix.with(
      this.#backend,
      this.outputSize,
      batches,
      type,
    );
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
      this.product,
      this.output,
      this.activationFn.activate(input.type),
    );
    return this.output;
  }

  // async backPropagate(): Promise<GPUMatrix> {
  //   await backPropagate(
  //     this.#backend,
  //     input,
  //     this.weights,
  //     this.product,
  //     this.output,
  //     this.activationFn.activate(input.type),
  //   );
  //   return this.output;
  // }

  toJSON() {
    return {
      outputSize: this.outputSize,
      activation: this.activationFn,
    };
  }
}
