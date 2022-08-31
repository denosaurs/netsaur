import { DataType, DataTypeArray } from "../../deps.ts";
import {
  Cost,
  DataSet,
  InputConfig,
  LayerConfig,
  Network,
  NetworkConfig,
} from "../types.ts";
import { fromType, getType } from "../util.ts";
import { CPUCostFunction, CrossEntropy, Hinge } from "./cost.ts";
import { BaseCPULayer } from "./layers/base.ts";
import { CPUMatrix } from "./matrix.ts";

export class CPUNetwork<T extends DataType = DataType> implements Network {
  input?: InputConfig;
  hidden: BaseCPULayer[];
  output: BaseCPULayer;
  silent: boolean;
  costFn: CPUCostFunction = new CrossEntropy();
  constructor(config: NetworkConfig) {
    this.silent = config.silent ?? false;
    this.input = config.input;
    this.hidden = config.hidden.map((layer) => new BaseCPULayer(layer));
    this.output = new BaseCPULayer(config.output);
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

  addLayers(layers: LayerConfig[]): void {
    this.hidden.push(...layers.map((layer) => new BaseCPULayer(layer)));
  }

  initialize(type: DataType, inputSize: number, batches: number) {
    this.hidden[0].initialize(type, inputSize, batches);

    for (let i = 1; i < this.hidden.length; i++) {
      const current = this.hidden[i];
      const previous = this.hidden[i - 1];
      current.initialize(type, previous.outputSize, batches);
    }

    const lastLayer = this.hidden[this.hidden.length - 1];
    this.output.initialize(type, lastLayer.outputSize, batches);
  }

  feedForward(input: CPUMatrix<T>): CPUMatrix<T> {
    for (const layer of this.hidden) {
      input = layer.feedForward(input);
    }
    input = this.output.feedForward(input);
    return input;
  }

  backpropagate(output: DataTypeArray<T>, rate: number) {
    const { x, y, type } = this.output.output;
    let error = CPUMatrix.with(x, y, type);
    for (const i in this.output.output.data) {
      error.data[i] = this.costFn.prime(
        output[i],
        this.output.output.data[i],
      );
    }
    this.output.backPropagate(error, rate);
    let weights = this.output.weights;
    for (let i = this.hidden.length - 1; i >= 0; i--) {
      error = CPUMatrix.dot(error, CPUMatrix.transpose(weights));
      this.hidden[i].backPropagate(error, rate);
      weights = this.hidden[i].weights;
    }
  }

  train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    rate: number,
  ): void {
    const type = this.input?.type ||
      getType(datasets[0].inputs as DataTypeArray<T>);
    const inputSize = this.input?.size || datasets[0].inputs.length / batches;

    this.initialize(type, inputSize, batches);

    if (!(datasets[0].inputs as DataTypeArray<T>).BYTES_PER_ELEMENT) {
      for (const dataset of datasets) {
        dataset.inputs = new (fromType(type))(dataset.inputs) as DataTypeArray<
          T
        >;
        dataset.outputs = new (fromType(type))(
          dataset.outputs,
        ) as DataTypeArray<T>;
      }
    }
    for (let e = 0; e < epochs; e++) {
      if (!this.silent) console.log(`Epoch ${e + 1}`);
      for (const dataset of datasets) {
        const input = new CPUMatrix(
          dataset.inputs as DataTypeArray<T>,
          inputSize,
          batches,
          type,
        );
        // TODO: do something with this output
        this.feedForward(input);
        this.backpropagate(dataset.outputs as DataTypeArray<T>, rate);
      }
    }
  }

  getCostLoss(output: DataTypeArray<T>) {
    const { x, y, type } = this.output.output;
    const cost = CPUMatrix.with(x, y, type);
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

  getOutput(): DataTypeArray<T> {
    return this.output.output.data as DataTypeArray<T>;
  }

  predict(data: DataTypeArray<T>) {
    const type = this.input?.type || getType(data);
    const input = new CPUMatrix(data, data.length, 1, type);
    for (const layer of this.hidden) {
      layer.reset(type, 1);
    }
    this.output.reset(type, 1);
    return this.feedForward(input).data;
  }

  toJSON() {
    return {
      type: "NeuralNetwork",
      sizes: this.hidden.map((layer) => layer.outputSize),
      input: this.input,
      hidden: this.hidden.map((layer) => layer.toJSON()),
      output: this.output.toJSON(),
    };
  }

  get weights(): CPUMatrix<T>[] {
    return this.hidden.map((layer) => layer.weights);
  }

  get biases(): CPUMatrix<T>[] {
    return this.hidden.map((layer) => layer.biases);
  }
}
