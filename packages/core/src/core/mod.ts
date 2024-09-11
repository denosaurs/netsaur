import {
  type Backend,
  Cost,
  type DataSet,
  LayerType,
  type NetworkConfig,
  SchedulerType,
} from "./types.ts";
import { Engine } from "./engine.ts";
import type { Rank } from "./api/shape.ts";
import type { Tensor } from "./tensor/tensor.ts";
import type { NeuralNetwork } from "./api/network.ts";
import { SGDOptimizer } from "./api/optimizer.ts";

/**
 * Sequential Neural Network
 */
export class Sequential implements NeuralNetwork {
  backend!: Backend;
  config: NetworkConfig;

  /**
   * Create a Sequential Neural Network.
   */
  constructor(config: NetworkConfig) {
    this.config = config;
    this.config.cost = this.config.cost || Cost.MSE;
    this.config.optimizer = this.config.optimizer || SGDOptimizer();
    this.config.scheduler = this.config.scheduler || {
      type: SchedulerType.None,
    };
    this.backend = Engine.backendLoader.loadBackend(this.config);
  }

  /**
   * Train the network using a set of training data.
   */
  train(datasets: DataSet[], epochs = 1000, batches = 1, rate = 0.1): void {
    this.backend.train(datasets, epochs, batches, rate);
  }

  /**
   * @param data
   * @param layers Range of layers [a, b) (inclusive of a, exclusive of b) to execute.
   * @returns
   */
  async predict(
    data: Tensor<Rank>,
    layers?: [number, number],
  ): Promise<Tensor<Rank>> {
    if (layers) {
      if (layers[0] < 0 || layers[1] > this.config.layers.length) {
        throw new RangeError(
          `Execution range should be within (0, ${this.config.layers.length}). Received (${(layers[
            0
          ],
            layers[1])})`,
        );
      }
      const lastLayer = this.config.layers[layers[1] - 1];
      const layerList = new Array(layers[1] - layers[0]);
      for (let i = 0; i < layerList.length; i += 1) {
        layerList[i] = layers[0] + i;
      }
      if (
        lastLayer.type === LayerType.Dense ||
        lastLayer.type === LayerType.Flatten
      ) {
        return await this.backend.predict(
          data,
          layerList,
          lastLayer.config.size,
        );
      } else if (lastLayer.type === LayerType.Activation) {
        const penultimate = this.config.layers[layers[1] - 2];
        if (
          penultimate.type === LayerType.Dense ||
          penultimate.type === LayerType.Flatten
        ) {
          return await this.backend.predict(
            data,
            layerList,
            penultimate.config.size,
          );
        } else {
          throw new Error(
            `The penultimate layer must be a dense layer, or a flatten layer if the last layer is an activation layer. Received ${penultimate.type}.`,
          );
        }
      } else {
        throw new Error(
          `The output layer must be a dense layer, activation layer, or a flatten layer. Received ${lastLayer.type}.`,
        );
      }
    }
    return await this.backend.predict(data);
  }

  /**
   * Load model from buffer
   */
  static load(data: Uint8Array): Sequential {
    return Engine.backendLoader.load(data);
  }

  /**
   * Load model from binary file
   */
  static loadFile(data: string): Sequential {
    return Engine.backendLoader.loadFile(data);
  }

  save(): Uint8Array {
    return this.backend.save();
  }

  saveFile(path: string): void {
    this.backend.saveFile(path);
  }
}
