import { DataTypeArray } from "../deps.ts";
import { CPUBackend } from "./cpu/backend.ts";
import {
  ConvLayerConfig,
  DataSet,
  DenseLayerConfig,
  Layer,
  Backend,
  NetworkConfig,
  NetworkJSON,
  PoolLayerConfig,
Activation,
LayerJSON,
} from "./types.ts";

/**
 * base class for neural network
 */
export class NeuralNetwork {
  backend!: Backend;
  /**
   * create a neural network
   */
  constructor(public config: NetworkConfig) {
    this.backend = new CPUBackend(this.config);
  }

  /**
   * setup backend and initialize network
   */
  async setupBackend(backendLoader: (config: NetworkConfig) => Promise<Backend>) {
    const backend = await backendLoader(this.config);
    this.backend = backend ?? new CPUBackend(this.config);
    return this;
  }

  /**
   * add layer to network
   */
  addLayer(layer: Layer) {
    this.backend.addLayer(layer);
  }

  /**
   * train network
   */
  async train(
    datasets: DataSet[],
    epochs = 1000,
    batches = 1,
    learningRate = 0.1,
  ) {
    await this.backend.train(datasets, epochs, batches, learningRate);
  }
  /**
   * use network to predict data
   */
  // deno-lint-ignore no-explicit-any
  async predict(data: DataTypeArray | any) {
    return await this.backend.predict(data);
  }

  /**
   * Export the network in a JSON format
   */
  toJSON(): NetworkJSON | undefined{
    return this.backend.toJSON();
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
  /**
   * get the weights of the network
   */
  getWeights() {
    return this.backend.getWeights();
  }

  /**
   * get the biases of the network
   */
  getBiases() {
    return this.backend.getBiases();
  }
}

export class DenseLayer {
  type = "dense";
  load = false;
  data?: LayerJSON;
  constructor(public config: DenseLayerConfig) {}
  static fromJSON(layerJSON: LayerJSON): DenseLayer {
    const layer = new DenseLayer({ size: layerJSON.outputSize, activation: (layerJSON.activationFn as Activation) || "sigmoid"});
    layer.load = true;
    layer.data = layerJSON;
    return layer;
  }
}

export class ConvLayer {
  type = "conv";
  load = false;
  data?: LayerJSON;
  constructor(public config: ConvLayerConfig) {}
  static fromJSON(layerJSON: LayerJSON): ConvLayer {
    const layer = new ConvLayer({ kernel: layerJSON.kernel!.data, kernelSize: { x: layerJSON.kernel!.x,  y: layerJSON.kernel!.y}, padding: layerJSON.padding, stride: layerJSON.stride});
    layer.load = true;
    layer.data = layerJSON;
    return layer;
  }
}

export class PoolLayer {
  type = "pool";
  load = false;
  data?: LayerJSON;
  constructor(public config: PoolLayerConfig) {}
  static fromJSON(layerJSON: LayerJSON): PoolLayer {
    const layer = new PoolLayer({stride: layerJSON.stride!});
    layer.load = true;
    layer.data = layerJSON;
    return layer;
  }
}
