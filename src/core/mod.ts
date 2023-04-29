import { Backend, BackendType, DataSet, NetworkConfig } from "./types.ts";
import { Data } from "../model/data/mod.ts";
import { Engine } from "./engine.ts";
import { Rank, Shape } from "./api/shape.ts";
import { Layer } from "./api/layer.ts";
import { Tensor } from "./tensor/tensor.ts";
import { NetworkJSON } from "../model/types.ts";

/**
 * base class for neural network
 */
export class NeuralNetwork {
  backend!: Backend;

  /**
   * create a neural network
   */
  constructor(public config: NetworkConfig) {
    this.backend = Engine.backendLoader.loadBackend(this.config);
  }

  /**
   * initialize the backend
   */
  initialize(inputSize: Shape[Rank], batches = 1) {
    this.backend.initialize(inputSize, batches);
  }

  /**
   * add layer to network
   */
  addLayer(layer: Layer) {
    this.backend.addLayer(layer);
    // this.layers.push(layer);
  }

  /**
   * train network
   */
  async train(
    datasets: (DataSet | Data)[],
    epochs = 1000,
    batches = 1,
    learningRate = 0.1,
  ) {
    await this.backend.train(datasets, epochs, batches, learningRate);
  }

  /**
   * use network to predict data
   */
  async predict(data: Tensor<Rank, BackendType>) {
    return await this.backend.predict(data);
  }

  /**
   * Export the network in a JSON format
   */
  async toJSON() {
    return await this.backend.toJSON();
  }

  /**
   * Import the network in a JSON format
   */
  static fromJSON(data: NetworkJSON) {
    return Engine.backendLoader.fromJSON(data);
  }

  /**
   * Load model from binary file
   */
  static load(_str: string) {
  }

  /**
   * save model to binary file
   */
  save(str: string) {
    this.backend.save(str);
  }
}