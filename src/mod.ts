import { DataType, DataTypeArray, WebGPUBackend, Core } from "../deps.ts";
import { CPUNetwork } from "./cpu/network.ts";
import { GPUNetwork } from "./gpu/network.ts";
import { DataSet, LayerConfig, Network, NetworkConfig } from "./types.ts";

/**
 * base class for neural network
 */
export class NeuralNetwork<T extends DataType = DataType> {
  network!: Network;
  /**
   * create a neural network
   */
  constructor(
    public config: NetworkConfig,
  ) {
    this.network = new CPUNetwork(this.config);
  }

  /**
   * setup backend and initialize network
   */
  async setupBackend(gpu = true) {
    const silent = this.config.silent;
    if (!gpu) {
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
   * add layers to network
   */
  addLayers(layer: LayerConfig[]) {
    this.network.addLayers(layer);
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
  getOutput() {
    return this.network.getOutput();
  }
  /**
   * use network to predict data
   */
  predict(data: DataTypeArray<T>) {
    return this.network.predict(data);
  }
}
