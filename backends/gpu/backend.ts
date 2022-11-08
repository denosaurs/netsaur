import { DenseGPULayer } from "./layers/dense.ts";
import type {
  Backend,
  BackendType,
  Cost,
  DataSet,
  GPULayer,
  GPUTensor,
  NetworkConfig,
  NetworkJSON,
  Rank,
  Shape,
  TensorData,
} from "../../core/types.ts";
import { CrossEntropy, GPUCostFunction, Hinge } from "./cost.ts";
import {
  DataTypeArray,
  WebGPUBackend,
} from "../../deps.ts";
import { GPUInstance } from "./mod.ts";
import { Tensor, toData } from "../../mod.ts";
import { flatten } from "../../core/util.ts";

export class GPUBackend implements Backend {
  input?: Shape[Rank];
  layers: GPULayer[] = [];
  output: GPULayer;
  silent: boolean;
  costFn: GPUCostFunction = new CrossEntropy();
  backend: WebGPUBackend;
  imported = false;

  constructor(config: NetworkConfig, backend: WebGPUBackend) {
    this.backend = backend;
    this.input = config.input;
    this.silent = config.silent ?? false;
    config.layers.map(this.addLayer.bind(this));
    this.output = config.layers.at(-1);
    this.setCost(config.cost);
  }

  setCost(activation: Cost): void {
    switch (activation) {
      case "crossentropy":
        this.costFn = new CrossEntropy();
        break;
      case "hinge":
        this.costFn = new Hinge();
        break;
    }
  }

  // deno-lint-ignore no-explicit-any
  addLayer(layer: any): void {
    this.layers.push(layer);
  }

  initialize(inputSize: Shape[Rank]) {
    this.layers[0].initialize(inputSize);
    for (let i = 1; i < this.layers.length; i++) {
      const current = this.layers[i];
      const previous = this.layers[i - 1];
      current.initialize(previous.output.shape);
    }
  }

  async feedForward(input: GPUTensor<Rank>) {
    for (const layer of this.layers) {
      input = await layer.feedForward(input);
    }
    return input;
  }

  async backpropagate(output: GPUTensor<Rank>, rate: number) {
    await this.output.backPropagate(
      output,
      output,
      rate,
      0,
      this.costFn,
    );
    // todo: update for convolutional layer
    for (let i = this.layers.length - 2, last = 1; i >= 0; i--, last++) {
      await this.layers[i].backPropagate(
        this.output.error,
        this.layers[i + 1].weights,
        rate,
        last,
      );
    }
  }

  async train(
    datasets: DataSet[],
    epochs = 5000,
    _batches = 1,
    rate = 0.1,
  ) {
    this.initialize(datasets[0].inputs.shape);

    for (let e = 0; e < epochs; e++) {
      if (!this.silent) console.log(`Epoch ${e + 1}/${epochs}`);
      for (const dataset of datasets) {
        await this.feedForward(dataset.inputs as GPUTensor<Rank>);
        await this.backpropagate(dataset.outputs as GPUTensor<Rank>, rate);
      }
    }
  }

  // async getCostLoss(_output: DataTypeArray<T>) {
  //   throw new Error("Not implemented");
  // }

  async predict(data: DataTypeArray) {
    const gpuData = toData(flatten(data as Float32Array)) as TensorData[BackendType.GPU];
    const input = new Tensor<Rank, BackendType.GPU>(gpuData, [data.length, 1]);
    this.layers.forEach((layer) => layer.reset(1));

    this.output.reset(1);
    return await (await this.feedForward(input)).data.get();
  }

  async toJSON(): Promise<NetworkJSON> {
    const layers = await Promise.all(
      this.layers.map(async (layer) => await layer.toJSON()),
    );
    return {
      costFn: this.costFn.name,
      sizes: this.layers.map((layer) => layer.outputSize),
      input: this.input,
      layers,
    };
  }

  static fromJSON(
    data: NetworkJSON,
    backend: WebGPUBackend,
  ): GPUBackend {
    if (!GPUInstance.backend) {
      throw new Error(
        "WebGPU backend not initialized, use loadBackend function",
      );
    }
    const layers = data.layers.map((layer) => {
      switch (layer.type) {
        case "dense":
          return DenseGPULayer.fromJSON(layer, GPUInstance.backend!);
        default:
          throw new Error(
            `${
              layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
            }Layer not implemented for the CPU backend`,
          );
      }
    });
    const gpubackend = new GPUBackend({
      input: data.input,
      layers: [],
      cost: data.costFn! as Cost,
    }, backend);
    gpubackend.layers = layers;
    gpubackend.output = layers.at(-1) as DenseGPULayer;
    return gpubackend;
  }

  save(_str: string): void {
    throw new Error("Not implemented");
  }

  getWeights(): GPUTensor<Rank>[] {
    return this.layers.map((layer) => (layer as DenseGPULayer).weights);
  }

  getBiases(): GPUTensor<Rank>[] {
    return this.layers.map((layer) => (layer as DenseGPULayer).biases);
  }
}
