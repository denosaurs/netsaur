import { BaseGPULayer } from "./layers/base.ts";
import {
  DataSet,
  InputConfig,
  LayerConfig,
  Network,
  NetworkConfig,
} from "../types.ts";
import { DataTypeArray, DataType, WebGPUBackend, WebGPUData } from "../../deps.ts";
import { fromType, getType } from "../util.ts";
import { GPUMatrix } from "./matrix.ts";
import { backPropagate } from "./kernels/backpropagate.ts";

export class GPUNetwork<T extends DataType = DataType> implements Network {
  input?: InputConfig;
  hidden: BaseGPULayer[];
  output: BaseGPULayer;
  backend: WebGPUBackend;
  silent: boolean;
  constructor(config: NetworkConfig, backend: WebGPUBackend) {
    this.silent = config.silent ?? false;
    this.input = config.input;
    this.backend = backend;
    this.output = new BaseGPULayer(config.output, backend);
    this.hidden = config.hidden.map((layer) =>
      new BaseGPULayer(layer, backend)
    );
  }

  addLayers(layers: LayerConfig[]) {
    this.hidden.push(
      ...layers.map((layer) => new BaseGPULayer(layer, this.backend)),
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

  async train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    learningRate: number,
  ) {
    const type = this.input?.type ||
      getType(datasets[0].inputs as DataTypeArray<T>);
    const inputSize = this.input?.size || datasets[0].inputs.length / batches;
    const outputSize = datasets[0].outputs.length / batches;

    for (let e = 0; e < epochs; e++) {
      for (const dataset of datasets) {
        await this.initialize(type, inputSize, batches);
        const inputArray = new (fromType(type))(dataset.inputs);
        const outputArray = new (fromType(type))(
          dataset.outputs,
        ) as DataTypeArray<
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
        await this.feedForward(input);

        // TODO loss function?

        await this.backpropagate(output, learningRate);
      }
    }
  }

  async predict(data: DataTypeArray<T>) {
    const type = this.input?.type || getType(data);
    const gpuData = await WebGPUData.from(this.backend, data);
    const input = new GPUMatrix<DataType>(gpuData, gpuData.length, 1, type);
    for (const layer of this.hidden) {
      await layer.reset(type, 1);
    }
    await this.output.reset(type, 1);
    return await (await this.feedForward(input)).data.get();
  }

  getOutput(): DataTypeArray<T> {
    return this.output.output.data as unknown as DataTypeArray<T>;
  }

  toJSON() {
    return {
      type: "NeuralNetwork",
      sizes: [
        this.input?.size,
        ...this.hidden.map((layer) => layer.outputSize),
      ],
      input: this.input,
      hidden: this.hidden.map((layer) => layer.toJSON()),
      output: this.output.toJSON(),
    };
  }

  get weights() {
    return [
      ...this.hidden.map((layer) => layer.weights),
      this.output.weights,
    ];
  }
}
