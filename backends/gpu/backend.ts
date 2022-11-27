import { DenseGPULayer } from "./layers/dense.ts";
import type {
  Backend,
  Cost,
  DataSet,
  GPULayer,
  GPUTensor,
  NetworkConfig,
  NetworkJSON,
  Rank,
  Shape,
} from "../../core/types.ts";
import { GPUCostFunction, MSE } from "./cost.ts";
import { WebGPUBackend } from "../../deps.ts";
import { GPUInstance } from "./mod.ts";

export class GPUBackend implements Backend {
  input?: Shape[Rank];
  layers: GPULayer[] = [];
  output: GPULayer;
  silent: boolean;
  costFn!: GPUCostFunction;
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
      case "mse":
        this.costFn = new MSE(this.backend);
        break;
    }
  }

  // deno-lint-ignore no-explicit-any
  addLayer(layer: any): void {
    this.layers.push(layer);
  }

  reset(size: Shape[Rank]) {
    for (const layer of this.layers) {
      size = layer.reset(size);
    }
    this.costFn.reset(size);
  }

  initialize(size: Shape[Rank]) {
    for (const layer of this.layers) {
      size = layer.initialize(size);
    }
    this.costFn.initialize(size);
  }

  async feedForward(input: GPUTensor<Rank>) {
    for (const layer of this.layers) {
      input = await layer.feedForward(input);
    }
    return input;
  }

  async backpropagate(output: GPUTensor<Rank>, rate: number) {
    await this.costFn.prime(this.output.output, output)
    let error = this.costFn.dInput
    for (let i = this.layers.length - 1; i >= 0; i--) {
      error = await this.layers[i].backPropagate(error, rate)!;
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
        if (!this.silent) {
          await this.getCostLoss(dataset.outputs as GPUTensor<Rank>);
          this.costFn.getOutput().then((loss) => console.log(`Loss ${loss}`));
        }
      }
    }
  }

  async getCostLoss(label: GPUTensor<Rank>) {
    await this.costFn.cost(this.output.output, label);
  }

  async predict(data: GPUTensor<Rank>) {
    data.shape.push(1);
    this.reset(data.shape);
    return await this.feedForward(data);
  }

  async toJSON(): Promise<NetworkJSON> {
    const layers = await Promise.all(
      this.layers.map(async (layer) => await layer.toJSON()),
    );
    return {
      costFn: this.costFn.name,
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
