import { Backend, BackendType, DataSet, NetworkConfig } from "./types.ts";
import { Engine } from "./engine.ts";
import { Rank } from "./api/shape.ts";
import { Tensor } from "./tensor/tensor.ts";
import { NetworkJSON } from "../model/types.ts";
import { NeuralNetwork } from "./api/network.ts";

/**
 * Sequential Neural Network
 */
export class Sequential implements NeuralNetwork {
  backend!: Backend;

  /**
   * Create a Sequential Neural Network.
   */
  constructor(public config: NetworkConfig) {
    this.backend = Engine.backendLoader.loadBackend(this.config);
  }

  /**
   * Train the Neural Network.
   */
  train(datasets: DataSet[], epochs = 1000, rate = 0.1) {
    this.backend.train(datasets, epochs, rate);
  }

  /**
   * Use the network to make predictions.
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
   * Save model to binary file
   */
  save(str: string) {
    this.backend.save(str);
  }
}
