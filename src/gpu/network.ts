import { GPULayer } from "./layer.ts";
import {
  DataSet,
  InputConfig,
  LayerConfig,
  Network,
  NetworkConfig,
} from "../types.ts";
import { DataArray, DataType, WebGPUBackend } from "../../deps.ts";
import { fromType, getType } from "../util.ts";
import { GPUMatrix } from "./matrix.ts";
import { backPropagate } from "./kernels/backpropagate.ts";

export class GPUNetwork<T extends DataType = DataType> implements Network {
  input?: InputConfig;
  hidden: GPULayer[];
  output: GPULayer;
  backend: WebGPUBackend;
  constructor(config: NetworkConfig, backend: WebGPUBackend) {
    this.input = config.input;
    this.backend = backend;
    this.output = new GPULayer(config.output, backend);
    this.hidden = config.hidden.map((layer) => new GPULayer(layer, backend));
  }

  addLayers(layers: LayerConfig[]) {
    this.hidden.push(
      ...layers.map((layer) => new GPULayer(layer, this.backend)),
    );
  }

  async initialize(type: DataType, inputSize: number, batches: number) {
    await this.hidden[0].initialize(type, inputSize, batches);

    for (let i = 1; i < this.hidden.length; i++) {
      const current = this.hidden[i];
      const previous = this.hidden[i - 1];
      await current.initialize(type, previous.outputSize, batches);
    }

    const lastLayer = this.hidden[this.hidden.length - 1];
    await this.output.initialize(type, lastLayer.outputSize, batches);
  }

  async feedForward(input: GPUMatrix) {
    for (const layer of this.hidden) {
      input = await layer.feedForward(input);
    }
    input = await this.output.feedForward(input);
    return input;
  }

  async backpropagate(output: GPUMatrix, learningRate: number) {
    const { x, y, type } = this.output.output;
    const error = await GPUMatrix.with(this.backend, x, y, type);
    const cost = await GPUMatrix.with(this.backend, x, y, type);
    await backPropagate(
      this.backend,
      this.output.output,
      output,
      error,
      cost,
      this.output.weights,
      this.output.inputs,
      learningRate,
      this.output.activationFn.prime(type),
      this.output.costFunction.prime(type),
    );
    return this.output.weights;
  }

  async train(datasets: DataSet[], epochs: number, batches: number) {
    const type = this.input?.type ||
      getType(datasets[0].inputs as DataArray<T>);
    const inputSize = this.input?.size || datasets[0].inputs.length / batches;
    const outputSize = datasets[0].outputs.length / batches;

    const inputArray = new (fromType(type))(datasets[0].inputs);
    const outputArray = new (fromType(type))(datasets[0].outputs) as DataArray<
      T
    >;

    const input = await GPUMatrix.from(
      this.backend,
      inputArray,
      inputSize,
      batches,
      type,
    );
    const output = await GPUMatrix.from(
      this.backend,
      outputArray,
      outputSize,
      batches,
      type,
    );

    for (let e = 0; e < epochs; e++) {
      await this.initialize(type, inputSize, batches);

      await this.feedForward(input);

      // TODO loss function?

      this.backpropagate(output, 0);
    }
  }

  getOutput(): DataArray<T> {
    throw new Error("Unimplemented!");
  }

  predict(_data: DataArray<T>): DataArray<T> {
    throw new Error("Unimplemented!");
  }
}
