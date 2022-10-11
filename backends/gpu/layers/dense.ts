import { DataType, WebGPUBackend, WebGPUData } from "../../../deps.ts";
import {
  Activation,
  DenseLayerConfig,
  LayerJSON,
  MatrixJSON,
  Size,
} from "../../../core/types.ts";
import { ActivationError, fromType, to1D } from "../../../core/util.ts";
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

/**
 * Base class for all layers.
 */
export class DenseGPULayer {
  outputSize: number;
  activationFn: GPUActivationFn = new Sigmoid();
  costFunction: GPUCostFunction = new CrossEntropy();

  input!: GPUMatrix;
  weights!: GPUMatrix;
  biases!: GPUMatrix;
  output!: GPUMatrix;
  error!: GPUMatrix;
  cost!: GPUMatrix;

  #backend: WebGPUBackend;

  constructor(config: DenseLayerConfig, backend: WebGPUBackend) {
    this.outputSize = to1D(config.size);
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

  async initialize(type: DataType, inputSize: Size, batches: number) {
    const b = this.#backend;
    const weights = new (fromType(type))(this.outputSize * to1D(inputSize))
      .map(() => Math.random() * 2 - 1);
    const biases = new (fromType(type))(this.outputSize)
      .map(() => Math.random() * 2 - 1);
    if (!this.weights) {
      this.weights = await GPUMatrix.with(
        b,
        this.outputSize,
        to1D(inputSize),
        type,
      );
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
    this.input = input;
    await feedForward(
      this.#backend,
      this.input,
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
    costFn: GPUCostFunction = this.costFunction,
  ): Promise<GPUMatrix> {
    await backPropagate(
      this.#backend,
      this.input,
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
      costFn.prime(error.type),
    );
    return this.output;
  }

  async toJSON(): Promise<LayerJSON> {
    const input = await this.input.toJSON();
    const weights = await this.weights.toJSON();
    const biases = await this.biases.toJSON();
    const output = await this.output.toJSON();
    const error = await this.error.toJSON();
    const cost = await this.cost.toJSON();

    return {
      outputSize: this.outputSize,
      activationFn: this.activationFn.name,
      costFn: this.costFunction.name,
      type: "dense",
      input,
      weights,
      biases,
      output,
      error,
      cost,
    };
  }

  static async fromJSON(
    { outputSize, activationFn, input, weights, biases, output }:
      LayerJSON,
    backend: WebGPUBackend,
  ): Promise<DenseGPULayer> {
    const layer = new DenseGPULayer({
      size: outputSize,
      activation: (activationFn as Activation) || "sigmoid",
    }, backend);
    layer.input = new GPUMatrix(
      await WebGPUData.from(backend, input.data, "f32"),
      input.x,
      input.y,
    );
    layer.weights = new GPUMatrix(
      await WebGPUData.from(backend, (weights as MatrixJSON).data, "f32"),
      (weights as MatrixJSON).x,
      (weights as MatrixJSON).y,
    );
    layer.biases = new GPUMatrix(
      await WebGPUData.from(backend, (biases as MatrixJSON).data, "f32"),
      (biases as MatrixJSON).x,
      (biases as MatrixJSON).y,
    );
    layer.output = new GPUMatrix(
      await WebGPUData.from(backend, output.data, "f32"),
      output.x,
      output.y,
    );
    return layer;
  }
}
