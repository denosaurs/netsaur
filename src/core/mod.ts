import { Backend, BackendType, DataSet, NetworkConfig } from "./types.ts";
import { Engine } from "./engine.ts";
import { Rank } from "./api/shape.ts";
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
   * train network
   */
  train(datasets: DataSet[], epochs = 1000, rate = 0.1) {
    this.backend.train(datasets, epochs, rate);
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
