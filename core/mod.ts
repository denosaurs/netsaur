import { DataTypeArray } from "../deps.ts";
import { CPUBackend } from "../backends/cpu/backend.ts";
import {
  Backend,
  DataSet,
  Layer,
  NetworkConfig,
  NetworkJSON,
Size,
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
  async setupBackend(
    backendLoader: (config: NetworkConfig) => Promise<Backend>,
  ) {
    const backend = await backendLoader(this.config);
    this.backend = backend ?? new CPUBackend(this.config);
    return this;
  }

  /**
   * initialize the backend
   */
  initialize(inputSize: Size, batches = 1) {
    this.backend.initialize(inputSize, batches);
  }

  /**
   * add layer to network
   */
  addLayer(layer: Layer) {
    this.backend.addLayer(layer);
  }

  /**
   * feed an input through the layers
   */
  // deno-lint-ignore no-explicit-any
  async feedForward(input: any): Promise<any> {
    return await this.backend.feedForward(input);
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
  async toJSON() {
    return await this.backend.toJSON();
  }
  /**
   * Import the network in a JSON format
   */
  static async fromJSON(
    data: NetworkJSON,
    helper?: (data: NetworkJSON, silent: boolean) => Promise<Backend>,
    silent = false,
  ) {
    return helper ? await helper(data, silent) : CPUBackend.fromJSON(data);
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
  /**
   * get layers from the backend
   */
  getLayer(index: number) {
    return this.backend.layers[index];
  }
}



