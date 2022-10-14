import { Backend, NetworkConfig, NetworkJSON } from "../../core/types.ts";

import { Core, WebGPUBackend } from "../../deps.ts";
import { GPUBackend } from "./backend.ts";
import { TensorGPUBackend } from "./tensor.ts";

export const GPU = {
  core: new Core(),
  initialized: false,
  init: async () => {
    if (GPU.initialized) {
      return GPU.core.backends.get("webgpu")! as WebGPUBackend;
    }
    await GPU.core.initialize();
    return GPU.core.backends.get("webgpu")! as WebGPUBackend;
  },

  backend: async (config: NetworkConfig): Promise<Backend> => {
    const silent = config.silent;
    const backend = await GPU.init();
    if (!backend.adapter) throw new Error("No backend adapter found!");
    if (!silent) console.log(`Using adapter: ${backend.adapter}`);
    const features = [...backend.adapter.features.values()];
    if (!silent) console.log(`Supported features: ${features.join(", ")}`);
    return new GPUBackend(config, backend);
  },
  model: async (data: NetworkJSON, silent = false): Promise<Backend> => {
    const backend = await GPU.init();
    if (!backend.adapter) throw new Error("No backend adapter found!");
    if (!silent) console.log(`Using adapter: ${backend.adapter}`);
    const features = [...backend.adapter.features.values()];
    if (!silent) console.log(`Supported features: ${features.join(", ")}`);
    const gpubackend = await GPUBackend.fromJSON(data, backend);
    console.log(gpubackend);
    return gpubackend;
  },
  tensor: async () => {
    const _backend = await GPU.init();
    return new TensorGPUBackend(_backend);
  },
};

export { GPUBackend };
export * from "./matrix.ts";
export type { DataSet } from "../../core/types.ts";
