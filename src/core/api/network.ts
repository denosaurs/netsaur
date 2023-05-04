import { Backend, BackendType, DataSet, NetworkConfig } from "../types.ts";
import { Tensor } from "../tensor/tensor.ts";
import { Rank } from "./shape.ts";

/**
 * Base Neural Network Structure. All Neural Networks should implement this.
 */
export interface NeuralNetwork {
  backend: Backend;

  config: NetworkConfig;

  /**
   * Train the Neural Network with the given datasets.
   * The number of epochs (default 1000) and the learning rate (default 0.1).
   */
  train(datasets: DataSet[], epochs?: number, rate?: number): void;

  /**
   * Use the network to make predictions on the given data.
   */
  predict(data: Tensor<Rank, BackendType>): Promise<Tensor<Rank, BackendType>>;

  /**
   * Save model to binary file
   */
  save(str: string): void;
}
