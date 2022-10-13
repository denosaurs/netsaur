import type { DataTypeArray } from "../../deps.ts";
import { ConvLayer, DenseLayer, PoolLayer } from "../../mod.ts";
import type {
  Backend,
  ConvLayerConfig,
  Cost,
  CPULayer,
  DataSet,
  DenseLayerConfig,
  Layer,
  NetworkConfig,
  NetworkJSON,
  PoolLayerConfig,
  Size,
} from "../../core/types.ts";
import { iterate1D, to1D } from "../../core/util.ts";
import { CPUCostFunction, CrossEntropy, Hinge } from "./cost.ts";
import { ConvCPULayer } from "./layers/conv.ts";
import { DenseCPULayer } from "./layers/dense.ts";
import { PoolCPULayer } from "./layers/pool.ts";
import { CPUMatrix } from "./matrix.ts";

type OutputLayer = DenseCPULayer;

export class CPUBackend implements Backend {
  input?: Size;
  layers: CPULayer[] = [];
  output: OutputLayer;
  silent: boolean;
  costFn: CPUCostFunction = new CrossEntropy();

  constructor(config: NetworkConfig) {
    this.input = config.input;
    this.silent = config.silent ?? false;
    config.layers.slice(0, -1).map(this.addLayer.bind(this));
    const output = config.layers[config.layers.length - 1];
    this.output = output.load
      ? DenseCPULayer.fromJSON(output.data!)
      : new DenseCPULayer(output.config as DenseLayerConfig);
    this.setCost(config.cost);
  }

  static load() {
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
          layer.load
            ? DenseCPULayer.fromJSON(layer.data!)
            : new DenseCPULayer(layer.config as DenseLayerConfig),
        );
        break;
      case "conv":
        this.layers.push(
          layer.load
            ? ConvCPULayer.fromJSON(layer.data!)
            : new ConvCPULayer(layer.config as ConvLayerConfig),
        );
        break;
      case "pool":
        this.layers.push(
          layer.load
            ? PoolCPULayer.fromJSON(layer.data!)
            : new PoolCPULayer(layer.config as PoolLayerConfig),
        );
        break;
      default:
        throw new Error(
          `${
            layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
          }Layer not implemented for the CPU backend`,
        );
    }
  }

  initialize(inputSize: Size, batches: number) {
    this.layers[0].initialize(inputSize, batches);

    for (let i = 1; i < this.layers.length; i++) {
      const current = this.layers[i];
      const previous = this.layers[i - 1];
      current.initialize(previous.outputSize, batches);
    }

    const lastLayer = this.layers[this.layers.length - 1];
    this.output.initialize(lastLayer.outputSize, batches);
  }

  feedForward(input: CPUMatrix): CPUMatrix {
    for (const layer of this.layers) {
      input = layer.feedForward(input);
    }
    input = this.output.feedForward(input);
    return input;
  }

  backpropagate(output: DataTypeArray, rate: number) {
    const { x, y } = this.output.output;
    let error = CPUMatrix.with(x, y);
    for (const i in this.output.output.data) {
      error.data[i] = this.costFn.prime(
        output[i],
        this.output.output.data[i],
      );
    }
    this.output.backPropagate(error, rate);
    // todo: update for convolutional layer
    let weights = (this.output as DenseCPULayer).weights;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      error = CPUMatrix.dot(error, CPUMatrix.transpose(weights));
      this.layers[i].backPropagate(error, rate);
      // todo: update for convolutional layer
      weights = (this.layers[i] as DenseCPULayer).weights;
    }
  }

  train(
    datasets: DataSet[],
    epochs = 5000,
    batches = 1,
    rate = 0.1,
  ): void {
    batches = datasets[0].inputs.tensor ? datasets[0].inputs.size.y : batches;
    const inputSize = datasets[0].inputs.tensor ? datasets[0].inputs.size.x : (this.input || datasets[0].inputs.length / batches);

    this.initialize(inputSize, batches ?? datasets[0].inputs.size.y);

    if (!(datasets[0].inputs as DataTypeArray).BYTES_PER_ELEMENT) {
      for (const dataset of datasets) {
        
        dataset.inputs = new Float32Array(dataset.inputs.tensor ? dataset.inputs.data : dataset.inputs);
        dataset.outputs = new Float32Array(dataset.outputs);
      }
    }
    iterate1D(epochs, (e: number) => {
      if (!this.silent) console.log(`Epoch ${e + 1}/${epochs}`);
      for (const dataset of datasets) {
        const input = new CPUMatrix(
          (dataset.inputs.tensor ? dataset.inputs.data : dataset.inputs) as DataTypeArray,
          to1D(inputSize),
          batches,
        );
        this.feedForward(input);
        this.backpropagate(dataset.outputs as DataTypeArray, rate);
      }
    });
  }

  getCostLoss(output: DataTypeArray) {
    const { x, y } = this.output.output;
    const cost = CPUMatrix.with(x, y);
    for (const i in this.output.output.data) {
      const activation = this.output.activationFn.prime(
        this.output.output.data[i],
      );
      cost.data[i] = activation * this.costFn.prime(
        output[i],
        this.output.output.data[i],
      );
    }
    return cost;
  }

  predict(data: DataTypeArray) {
    const input = new CPUMatrix(data, data.length, 1);
    for (const layer of this.layers) {
      layer.reset(1);
    }
    this.output.reset(1);
    return this.feedForward(input).data;
  }

  toJSON(): NetworkJSON {
    return {
      costFn: this.costFn.name,
      type: "NeuralNetwork",
      sizes: this.layers.map((layer) => layer.outputSize),
      input: this.input,
      layers: this.layers.map((layer) => layer.toJSON()),
      output: this.output.toJSON(),
    };
  }

  static fromJSON(data: NetworkJSON): CPUBackend {
    const layers = data.layers.map((layer) => {
      switch (layer.type) {
        case "dense":
          return DenseLayer.fromJSON(layer);
        case "conv":
          return ConvLayer.fromJSON(layer);
        case "pool":
          return PoolLayer.fromJSON(layer);
        default:
          throw new Error(
            `${
              layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
            }Layer not implemented for the CPU backend`,
          );
      }
    });
    layers.push(DenseLayer.fromJSON(data.output));
    const backend = new CPUBackend({
      input: data.input,
      layers,
      cost: data.costFn! as Cost,
    });
    return backend;
  }

  save(_str: string): void {
    throw new Error("Not implemented");
  }

  getWeights(): CPUMatrix[] {
    return this.layers.map((layer) => (layer as DenseCPULayer).weights);
  }

  getBiases(): CPUMatrix[] {
    return this.layers.map((layer) => (layer as DenseCPULayer).biases);
  }
}
