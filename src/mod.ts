import { WebGPUBackend } from "../deps.ts";
import { CPUNetwork } from "./cpu/network.ts";
import { GPUNetwork } from "./gpu/network.ts";
import { LayerConfig, Network, NetworkConfig } from "./types.ts";

export class NeuralNetwork {
  public config: NetworkConfig;

  public network!: Network;

  constructor(config: NetworkConfig) {
    this.config = config;
  }

  public async setupBackend(gpu = true) {
    if (!gpu) return this.network = new CPUNetwork(this.config);

    const backend = new WebGPUBackend();
    await backend.initialize()
    if (backend.adapter) {
      console.log(`Using adapter: ${backend.adapter.name}`);
      const features = [...backend.adapter.features.values()];
      console.log(`Supported features: ${features.join(", ")}`);

      this.network = new GPUNetwork(this.config, backend);
    } else {
      console.error("No adapter found");
      this.network = new CPUNetwork(this.config);
    }
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

  public async train() {
    if (!this.network) await this.setupBackend()
  }

  public async predict() {
    if (!this.network) await this.setupBackend()
  }
}
