import { Core, DataTypeArray, WebGPUBackend } from "../deps.ts";
import { CPUNetwork } from "./cpu/network.ts";
import { GPUNetwork } from "./gpu/network.ts";
import {
  Backend,
  ConvLayerConfig,
  DataSet,
  DenseLayerConfig,
  Layer,
  Network,
  NetworkConfig,
  NetworkJSON,
  PoolLayerConfig,
} from "./types.ts";

/**
 * base class for neural network
 */
export class NeuralNetwork {
  network!: Network;
  /**
   * create a neural network
   */
  constructor(public config: NetworkConfig) {
    this.network = new CPUNetwork(this.config);
  }

  /**
   * setup backend and initialize network
   */
  async setupBackend(backendType: Backend | boolean = false) {
    const silent = this.config.silent;
    if (!backendType || backendType === "CPU" || backendType === "cpu") {
      this.network = new CPUNetwork(this.config);
      return this;
    }
    const core = new Core();
    await core.initialize();
    const backend = core.backends.get("webgpu")! as WebGPUBackend;
    if (backend.adapter) {
      if (!silent) console.log(`Using adapter: ${backend.adapter}`);
      const features = [...backend.adapter.features.values()];
      if (!silent) console.log(`Supported features: ${features.join(", ")}`);

      this.network = new GPUNetwork(this.config, backend);
    } else {
      console.error("No adapter found");
      this.network = new CPUNetwork(this.config);
    }

    return this;
  }

  // public withDevice(adapter: GPUAdapter, device: GPUDevice) {
  //   console.log(`Using adapter: ${adapter.name}`);
  //   const features = [...adapter.features.values()];
  //   console.log(`Supported features: ${features.join(", ")}`);
  //   this.network = new GPUNetwork(this.config);
  // }

  /**
   * add layer to network
   */
  addLayer(layer: Layer) {
    this.network.addLayer(layer);
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
    await this.network.train(datasets, epochs, batches, learningRate);
  }

  /**
   * get output of network
   */
  async getOutput() {
    return await this.network.getOutput();
  }

  /**
   * use network to predict data
   */
  async predict(data: DataTypeArray) {
    return await this.network.predict(data);
  }

  /**
   * Export the network in a JSON format
   */
  toJSON(): NetworkJSON {
    return this.network.toJSON();
  }

  /**
   * get the weights of the network
   */
  getWeights() {
    return this.network.getWeights();
  }

  /**
   * get the biases of the network
   */
  getBiases() {
    return this.network.getBiases();
  }
}

export class DenseLayer {
  public type = "dense";
  constructor(public config: DenseLayerConfig) {}
}

export class ConvLayer {
  public type = "conv";
  constructor(public config: ConvLayerConfig) {}
}

export class PoolLayer {
  public type = "pool";
  constructor(public config: PoolLayerConfig) {}
}
