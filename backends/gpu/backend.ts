import { DenseGPULayer } from "./layers/dense.ts";
import type {
  Backend,
  Cost,
  DataSet,
  GPULayer,
  Layer,
  NetworkConfig,
  NetworkJSON,
  Size,
} from "../../core/types.ts";
import { CrossEntropy, GPUCostFunction, Hinge } from "./cost.ts";
import {
  DataType,
  DataTypeArray,
  WebGPUBackend,
  WebGPUData,
} from "../../deps.ts";
import { fromType, getType, to1D } from "../../core/util.ts";
import { GPUMatrix } from "./matrix.ts";
import { DenseLayer } from "../../mod.ts";

export class GPUBackend<T extends DataType = DataType> implements Backend {
  input?: Size;
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
    config.layers.slice(0, -1).map(this.addLayer.bind(this));
    this.output = new DenseGPULayer(
      (config.layers.at(-1) as DenseLayer)!.config,
      backend,
    );
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

  addLayer(layer: Layer): void {
    switch (layer.type) {
      case "dense":
        this.layers.push(
          new DenseGPULayer((layer as DenseLayer).config, this.backend),
        );
        break;
      case "conv":
        throw new Error(`ConvLayer not implemented for the GPU backend`);
      default:
        throw new Error(
          `${
            layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
          }Layer not implemented for the GPU backend`,
        );
    }
  }

  async initialize(inputSize: Size, batches: number, type: DataType) {
    await this.layers[0].initialize(type, inputSize, batches);
    for (let i = 1; i < this.layers.length; i++) {
      const current = this.layers[i];
      const previous = this.layers[i - 1];
      await current.initialize(
        type,
        previous.outputSize,
        batches,
      );
    }

    const lastLayer = this.layers[this.layers.length - 1];
    await this.output.initialize(
      type,
      lastLayer.outputSize,
      batches,
    );
  }

  async feedForward(input: GPUMatrix) {
    for (const layer of this.layers) {
      input = await layer.feedForward(input);
    }
    input = await this.output.feedForward(input);
    return input;
  }

  async backpropagate(output: GPUMatrix, rate: number) {
    await this.output.backPropagate(
      output,
      output,
      rate,
      0,
      this.costFn,
    );
    // todo: update for convolutional layer
    let weights = this.output.weights;
    let last = 1;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      await this.layers[i].backPropagate(
        this.output.error,
        weights,
        rate,
        last,
      );
      // todo: update for convolutional layer
      weights = this.layers[i].weights;
      last++;
    }
  }

  async train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    rate: number,
  ) {
    const type = getType(datasets[0].inputs as DataTypeArray<T>);
    const inputSize = this.input || datasets[0].inputs.length / batches;
    const outputSize = datasets[0].outputs.length / batches;

    await this.initialize(inputSize, batches, type);

    const databuffers = [];
    for (const dataset of datasets) {
      const inputArray = new (fromType(type))(dataset.inputs);
      const outputArray = new (fromType(type))(dataset.outputs);

      const input = await GPUMatrix.from(
        this.backend,
        inputArray,
        to1D(inputSize),
        batches,
      );
      const output = await GPUMatrix.from(
        this.backend,
        outputArray,
        outputSize,
        batches,
      );

      databuffers.push({ input, output });
    }

    for (let e = 0; e < epochs; e++) {
      if (!this.silent) console.log(`Epoch ${e + 1}/${epochs}`);
      for (const dataset of databuffers) {
        await this.feedForward(dataset.input);
        await this.backpropagate(dataset.output, rate);
      }
    }
  }

  // deno-lint-ignore require-await
  async getCostLoss(_output: DataTypeArray<T>) {
    throw new Error("Not implemented");
  }

  async predict(data: DataTypeArray<T>) {
    const type = getType(data);
    const gpuData = await WebGPUData.from(this.backend, data);
    const input = new GPUMatrix<DataType>(gpuData, gpuData.length, 1, type);
    this.layers.forEach(async (layer) => {
      await layer.reset(type, 1);
    });

    await this.output.reset(type, 1);
    return await (await this.feedForward(input)).data.get();
  }

  async toJSON(): Promise<NetworkJSON> {
    const layers = await Promise.all(
      this.layers.map(async (layer) => await layer.toJSON()),
    );
    return {
      costFn: this.costFn.name,
      type: "NeuralNetwork",
      sizes: this.layers.map((layer) => layer.outputSize),
      input: this.input,
      layers,
      output: await this.output.toJSON(),
    };
  }

  static async fromJSON(
    data: NetworkJSON,
    backend: WebGPUBackend,
  ): Promise<GPUBackend> {
    const layers = data.layers.map((layer) => {
      switch (layer.type) {
        case "dense":
          return DenseLayer.fromJSON(layer);
        default:
          throw new Error(
            `${
              layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
            }Layer not implemented for the CPU backend`,
          );
      }
    });
    layers.push(DenseLayer.fromJSON(data.output));
    const gpubackend = new GPUBackend({
      input: data.input,
      layers,
      cost: data.costFn! as Cost,
    }, backend);
    gpubackend.output = await DenseGPULayer.fromJSON(
      (layers.at(-1) as DenseLayer)!.data!,
      backend,
    );
    layers.slice(0, -1).forEach(async (layer) => {
      if (layer.type === "dense") {
        gpubackend.layers.push(
          await DenseGPULayer.fromJSON(layer.data!, backend),
        );
      } else {
        throw new Error(
          `${
            layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
          }Layer not implemented for the GPU backend`,
        );
      }
    });
    return gpubackend;
  }

  save(_str: string): void {
    throw new Error("Not implemented");
  }

  getWeights(): GPUMatrix[] {
    return this.layers.map((layer) => (layer as DenseGPULayer).weights);
  }

  getBiases(): GPUMatrix[] {
    return this.layers.map((layer) => (layer as DenseGPULayer).biases);
  }
}
