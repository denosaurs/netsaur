import {
  Backend,
  Cost,
  DataSet,
  LayerType,
  NetworkConfig,
  SchedulerType,
} from "./types.ts";
import { Engine } from "./engine.ts";
import { Rank } from "./api/shape.ts";
import { Tensor } from "./tensor/tensor.ts";
import { NeuralNetwork } from "./api/network.ts";
import { SGDOptimizer } from "./api/optimizer.ts";
import { Shape } from "./api/shape.ts";
import { DenseLayer } from "../../mod.ts";

/**
 * Sequential Neural Network
 */
export class Sequential implements NeuralNetwork {
  backend!: Backend;

  /**
   * Create a Sequential Neural Network.
   */
  constructor(public config: NetworkConfig) {
    this.config.cost = this.config.cost || Cost.MSE;
    this.config.optimizer = this.config.optimizer || SGDOptimizer();
    this.config.scheduler = this.config.scheduler || {
      type: SchedulerType.None,
    };
    this.backend = Engine.backendLoader.loadBackend(this.config);
  }

  train(datasets: DataSet[], epochs = 1000, batches = 1, rate = 0.1) {
    this.backend.train(datasets, epochs, batches, rate);
  }

  async predict(data: Tensor<Rank>, layers?: [number, number]) {
    if (layers) {
      if (layers[0] < 0 || layers[1] > this.config.layers.length)
        throw new RangeError(
          `Execution range should be within (0, ${
            this.config.layers.length
          }). Received (${(layers[0], layers[1])})`
        );
      const lastLayer = this.config.layers[layers[1] - 1];
      const layerList = new Array(layers[1] - layers[0]);
      for (let i = 0; i < layerList.length; i += 1) {
        layerList[i] = layers[0] + i;
      }
      if (
        lastLayer.type === LayerType.Dense ||
        lastLayer.type === LayerType.Flatten
      ) {
        return await this.backend.predict(data, layerList, lastLayer.config.size);
      } else if (lastLayer.type === LayerType.Activation) {
        const penultimate = this.config.layers[layers[1] - 2];
        if (
          penultimate.type === LayerType.Dense ||
          penultimate.type === LayerType.Flatten
        ) {
          return await this.backend.predict(
            data,
            layerList,
            penultimate.config.size
          );
        } else {
          throw new Error(
            `The penultimate layer must be a dense layer, or a flatten layer if the last layer is an activation layer. Received ${penultimate.type}.`
          );
        }
      } else {
        throw new Error(
          `The output layer must be a dense layer, activation layer, or a flatten layer. Received ${lastLayer.type}.`
        );
      }
    }
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
