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

  /**
   * Train the Neural Network.
   */
  train(datasets: DataSet[], epochs = 1000, rate = 0.1) {
    this.backend.train(datasets, epochs, rate);
  }

  /**
   * Use the network to make predictions.
   */
  async predict(data: Tensor<Rank>) {
    return await this.backend.predict(data);
  }

  /**
   * Load model from binary file
   */
  static loadModel(path: string): Sequential;
  static loadModel(data: Uint8Array): Sequential;
  static loadModel(_data: string | Uint8Array) {
    return null as unknown as Sequential;
  }

  /**
   * Save model to binary file
   */
  save(str: string) {
    this.backend.save(str);
  }
}
