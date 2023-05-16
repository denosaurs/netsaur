import { Backend, DataSet, NetworkConfig } from "./types.ts";
import { Engine } from "./engine.ts";
import { Rank } from "./api/shape.ts";
import { Tensor } from "./tensor/tensor.ts";
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

  train(datasets: DataSet[], epochs = 1000, batches = 1, rate = 0.1) {
    this.backend.train(datasets, epochs, batches, rate);
  }

  async predict(data: Tensor<Rank>) {
    return await this.backend.predict(data);
  }

  /**
   * Load model from buffer
   */
  static load(data: Uint8Array) {
    return Engine.backendLoader.load(data);
  }

  /**
   * Load model from binary file
   */
  static loadFile(data: string) {
    return Engine.backendLoader.loadFile(data);
  }

  save(): Uint8Array {
    return this.backend.save();
  }

  saveFile(path: string) {
    this.backend.saveFile(path);
  }
}
