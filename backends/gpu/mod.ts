import { Backend, NetworkConfig, NetworkJSON } from "../../core/types.ts";

import { Core, WebGPUBackend } from "../../deps.ts";
import { GPUBackend } from "./backend.ts";
import { TensorGPUBackend } from "./tensor.ts";


class GPUInstance {
  static core = new Core();
  static backend?: WebGPUBackend;
  static initialized = false;

  static async init(silent = false): Promise<void> {
    if (GPUInstance.initialized) return;
    await GPUInstance.core.initialize();
    GPUInstance.backend = GPUInstance.core.backends.get("webgpu")! as WebGPUBackend;
    GPUInstance.initialized = true;
    if (!GPUInstance.backend.adapter) throw new Error("No backend adapter found!");
    if (!silent) console.log(`Using adapter: ${GPUInstance.backend.adapter}`);
    const features = [...GPUInstance.backend.adapter.features.values()];
    if (!silent) console.log(`Supported features: ${features.join(", ")}`);
  }
}
export const GPU = {
  backend: async (config: NetworkConfig): Promise<Backend> => {
    await GPUInstance.init(config.silent);
    return new GPUBackend(config, GPUInstance.backend!);
  },
  model: async (data: NetworkJSON, silent = false): Promise<Backend> => {
    await GPUInstance.init(silent);
    const gpubackend = await GPUBackend.fromJSON(data, GPUInstance.backend!);
    return gpubackend;
  },
  tensor: async () => {
    await GPUInstance.init(true);
    return new TensorGPUBackend(GPUInstance.backend!);
  },
};

export { GPUBackend };
export * from "./matrix.ts";
export type { DataSet } from "../../core/types.ts";
