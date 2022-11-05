import type {
  Backend,
  Cost,
  CPULayer,
  CPUTensor,
  DataSet,
  NetworkConfig,
  NetworkJSON,
  Rank,
  Shape,
} from "../../core/types.ts";
import { iterate1D } from "../../core/util.ts";
import { CPUCostFunction, CrossEntropy, Hinge } from "./cost.ts";
import { ConvCPULayer } from "./layers/conv.ts";
import { DenseCPULayer } from "./layers/dense.ts";
import { PoolCPULayer } from "./layers/pool.ts";
import { cpuZeroes2D } from "../../mod.ts";

type OutputLayer = DenseCPULayer;

export class CPUBackend implements Backend {
  input?: Shape[Rank];
  layers: CPULayer[] = [];
  output: OutputLayer;
  silent: boolean;
  costFn: CPUCostFunction = new CrossEntropy();

  constructor(config: NetworkConfig) {
    this.input = config.input;
    this.silent = config.silent ?? false;
    config.layers.map(this.addLayer.bind(this));
    this.output = config.layers.at(-1);
    this.setCost(config.cost);
  }

  static load(path: string) {
    const net = JSON.parse(Deno.readTextFileSync(path))
    return CPUBackend.fromJSON(net)
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

  initialize(shape: Shape[Rank]) {
    this.layers[0]?.initialize(shape);

    for (let i = 1; i < this.layers.length; i++) {
      const current = this.layers[i];
      const previous = this.layers[i - 1];
      current.initialize(previous.output.shape);
    }
  }

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    for (const layer of this.layers) {
      input = layer.feedForward(input);
    }
    return input;
  }

  backpropagate(output: CPUTensor<Rank>, rate: number) {
    const shape = this.output.output.shape;
    let error = cpuZeroes2D(shape) as CPUTensor<Rank>;
    for (const i in this.output.output.data) {
      error.data[i] = this.costFn.prime(
        output.data[i],
        this.output.output.data[i],
      );
    }
    for (let i = this.layers.length - 1; i >= 0; i--) {
      error = this.layers[i].backPropagate(error, rate)!;
    }
  }

  train(
    datasets: DataSet[],
    epochs = 5000,
    _batches = 1,
    rate = 0.1,
  ): void {
    this.initialize(datasets[0].inputs.shape);

    iterate1D(epochs, (e: number) => {
      if (!this.silent) console.log(`Epoch ${e + 1}/${epochs}`);
      for (const dataset of datasets) {
        this.feedForward(dataset.inputs as CPUTensor<Rank>);
        this.backpropagate(dataset.outputs as CPUTensor<Rank>, rate);
      }
    });
  }

  // getCostLoss(output: DataTypeArray) {
  //   const { x, y } = this.output.output;
  //   const cost = CPUMatrix.with(x, y);
  //   for (const i in this.output.output.data) {
  //     const activation = this.output.activationFn.prime(
  //       this.output.output.data[i],
  //     );
  //     cost.data[i] = activation * this.costFn.prime(
  //       output[i],
  //       this.output.output.data[i],
  //     );
  //   }
  //   return cost;
  // }

  predict(input: CPUTensor<Rank>) {
    for (const layer of this.layers) {
      layer.reset(1);
    }
    return this.feedForward(input);
  }

  async toJSON() {
    return {
      costFn: this.costFn.name,
      sizes: this.layers.map((layer) => layer.outputSize),
      input: this.input,
      layers: await Promise.all(
        this.layers.map(async (layer) => await layer.toJSON()),
      ),
    };
  }

  static fromJSON(data: NetworkJSON): CPUBackend {
    const layers = data.layers.map((layer) => {
      switch (layer.type) {
        case "dense":
          return DenseCPULayer.fromJSON(layer);
        case "conv":
          return ConvCPULayer.fromJSON(layer);
        case "pool":
          return PoolCPULayer.fromJSON(layer);
        default:
          throw new Error(
            `${
              layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
            }Layer not implemented for the CPU backend`,
          );
      }
    });
    const backend = new CPUBackend({
      input: data.input,
      layers: [],
      cost: data.costFn! as Cost,
    });
    backend.layers = layers
    backend.output = layers.at(-1) as DenseCPULayer
    return backend;
  }

  async save(path: string) {
    const data = await this.toJSON()
    Deno.writeTextFileSync(path, JSON.stringify(data))
  }

  getWeights(): CPUTensor<Rank>[] {
    return this.layers.map((layer) => (layer as DenseCPULayer).weights);
  }

  getBiases(): CPUTensor<Rank>[] {
    return this.layers.map((layer) => (layer as DenseCPULayer).biases);
  }
}
