import { DataArray, DataType } from "../../deps.ts";
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
import { CPULayer } from "./layer.ts";
import { CPUMatrix } from "./matrix.ts";

export class CPUNetwork<T extends DataType = DataType> implements Network {
  input?: InputConfig;
  hidden: CPULayer[];
  output: CPULayer;

  constructor(config: NetworkConfig) {
    this.input = config.input;
    this.hidden = config.hidden.map((layer) => new CPULayer(layer));
    this.output = new CPULayer(config.output);
    this.setCost(config.cost);
  }

  setCost(activation: Cost): void {
    let costFn: CPUCostFunction;
    switch (activation) {
      case "crossentropy":
        costFn = new CrossEntropy();
        break;
      case "hinge":
        costFn = new Hinge();
        break;
    }
    this.hidden.map((layer) => layer.costFn = costFn);
    this.output.costFn = costFn;
  }

  addLayers(layers: LayerConfig[]): void {
    this.hidden.push(...layers.map((layer) => new CPULayer(layer)));
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

  backpropagate(output: DataArray<T>, learningRate: number) {
    const { x, y, type } = this.output.output;
    let error = CPUMatrix.with(x, y, type);
    const cost = CPUMatrix.with(x, y, type);
    for (const i in this.output.output.data) {
      const activation = this.output.activationFn.prime(
        this.output.output.data[i],
      );
      error.data[i] = this.output.costFn.prime(
        output[i],
        this.output.output.data[i],
      );
      cost.data[i] = activation * error.data[i];
    }
    const weightsDelta = CPUMatrix.dot(
      CPUMatrix.transpose(this.output.input),
      cost,
    );
    for (const i in weightsDelta.data) {
      this.output.weights.data[i] += weightsDelta.data[i] * 1;
    }
    console.log(this.output.weights);
    for (let i = 0, j = 0; i < cost.data.length; i++, j++) {
      if (j >= this.output.biases.x) j = 0;
      this.output.biases.data[j] += cost.data[i] * learningRate;
    }
    const lastLayer = this.hidden[this.hidden.length - 1];
    error = lastLayer.backPropagate(error, this.output.weights, learningRate);
    for (let i = this.hidden.length - 2; i >= 0; i--) {
      const prevLayer = this.hidden[i + 1];
      error = this.hidden[i].backPropagate(
        error,
        prevLayer.weights,
        learningRate,
      );
    }
  }

  train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    learningRate: number,
  ): void {
    const type = this.input?.type ||
      getType(datasets[0].inputs as DataArray<T>);
    const inputSize = this.input?.size || datasets[0].inputs.length / batches;

    this.initialize(type, inputSize, batches);

    if (!(datasets[0].inputs as DataArray<T>).BYTES_PER_ELEMENT) {
      for (const dataset of datasets) {
        dataset.inputs = new (fromType(type))(dataset.inputs) as DataArray<T>;
        dataset.outputs = new (fromType(type))(dataset.outputs) as DataArray<T>;
      }
    }
    for (let e = 0; e < epochs; e++) {
      for (const dataset of datasets) {
        const input = new CPUMatrix(
          dataset.inputs as DataArray<T>,
          inputSize,
          batches,
          type,
        );
        // TODO: do something with this output
        this.feedForward(input);
        this.backpropagate(dataset.outputs as DataArray<T>, learningRate);
      }
    }
  }

  get weights(): CPUMatrix<T>[] {
    return this.hidden.map((layer) => layer.weights);
  }
  get biases(): CPUMatrix<T>[] {
    return this.hidden.map((layer) => layer.biases);
  }

  getOutput(): DataArray<T> {
    return this.output.output.data as DataArray<T>;
  }

  predict(data: DataArray<T>) {
    const type = this.input?.type || getType(data);
    const input = new CPUMatrix(data, data.length, 1, type);
    for (const layer of this.hidden) {
      layer.reset(type, 1);
    }
    this.output.reset(type, 1);
    return this.feedForward(input).data;
  }
}
