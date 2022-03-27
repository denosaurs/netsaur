import { DataType, WebGPUBackend } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { fromType } from "../util.ts";
import {
  GPUActivationFn,
  LeakyRelu,
  Relu,
  Sigmoid,
  Tanh,
} from "./activation.ts";
import { CrossEntropy, GPUCostFunction } from "./cost.ts";
import { matMul } from "./kernels/matmul.ts";
import { GPUMatrix } from "./matrix.ts";

interface GPULayerConfig extends LayerConfig {
  size: number;
  activation: Activation;
}

export class GPULayer<T extends DataType = DataType> {
  outputSize: number;
  activationFn: GPUActivationFn = new Sigmoid();
  costFunction: GPUCostFunction = new CrossEntropy();

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
    }
  }

  // TODO: memoization
  async feedForward(input: GPUMatrix): Promise<GPUMatrix> {
    await matMul(
      this.#backend,
      input,
      this.weights,
      this.output,
      this.activationFn.activate(input.type),
    );
    return this.output;
  }
}
