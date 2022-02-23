import { DataType, WebGPUBackend } from "../deps.ts";
import { CPUNetwork } from "./cpu/network.ts";
// import { GPUNetwork } from "./gpu/network.ts";
import { DataSet, LayerConfig, Network, NetworkConfig } from "./types.ts";

export class NeuralNetwork<T extends DataType = DataType> {
  public config: NetworkConfig;

  public network!: Network;

  constructor(config: NetworkConfig) {
    this.config = config;
  }

  // deno-lint-ignore require-await
  public async setupBackend(gpu = true, silent = true) {
    if (!gpu) {
      this.network = new CPUNetwork(this.config);
      return this
    }
    /*const backend = new WebGPUBackend();
    await backend.initialize()
    if (backend.adapter) {
      if (!silent) console.log(`Using adapter: ${backend.adapter.name}`);
      const features = [...backend.adapter.features.values()];
      if (!silent) console.log(`Supported features: ${features.join(", ")}`);

      this.network = new GPUNetwork(this.config, backend);
    } else {
      console.error("No adapter found");
      this.network = new CPUNetwork(this.config);
    }

    return this*/
  }

  // public withDevice(adapter: GPUAdapter, device: GPUDevice) {
  //   console.log(`Using adapter: ${adapter.name}`);
  //   const features = [...adapter.features.values()];
  //   console.log(`Supported features: ${features.join(", ")}`);
  //   this.network = new GPUNetwork(this.config);
  // }

  public addLayers(layer: LayerConfig[]) {
    this.network.addLayers(layer)
  }

  public async train(datasets: DataSet<T>, epochs = 1000, batches = 1) {
    if (!this.network) await this.setupBackend()
    this.network.train(datasets, epochs, batches)
  }

  public async predict() {
    if (!this.network) await this.setupBackend()
  }
}
