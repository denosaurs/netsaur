import { Engine } from "../../core/engine.ts";
import { Tensor } from "../../core/tensor.ts";
import {
  Backend,
  DenseLayerConfig,
  NetworkConfig,
  NetworkJSON,
} from "../../core/types.ts";

import { Core, WebGPUBackend } from "../../deps.ts";
import { Layer } from "../../layers/mod.ts";
import { GPUBackend } from "./backend.ts";
import { DenseGPULayer } from "./layers/dense.ts";
import { TensorGPUBackend } from "./tensor.ts";

export class GPUInstance {
  static core = new Core();
  static backend?: WebGPUBackend;
  static initialized = false;

  static async init(silent = false): Promise<void> {
    if (GPUInstance.initialized) return;
    await GPUInstance.core.initialize();
    GPUInstance.backend = GPUInstance.core.backends.get(
      "webgpu",
    )! as WebGPUBackend;
    GPUInstance.initialized = true;
    if (!GPUInstance.backend.adapter) {
      throw new Error("No backend adapter found!");
    }
    if (!silent) console.log(`Using adapter: ${GPUInstance.backend.adapter}`);
    const features = [...GPUInstance.backend.adapter.features.values()];
    if (!silent) console.log(`Supported features: ${features.join(", ")}`);
  }
}

const loadBackend = (config: NetworkConfig): Backend => {
  if (!GPUInstance.backend) {
    throw new Error("WebGPU backend not initialized, use loadBackend function");
  }
  return new GPUBackend(config, GPUInstance.backend!);
};

const model = async (data: NetworkJSON, silent = false): Promise<Backend> => {
  await GPUInstance.init(silent);
  const gpubackend = await GPUBackend.fromJSON(data, GPUInstance.backend!);
  return gpubackend;
};

const dense = (config: DenseLayerConfig) => {
  if (!GPUInstance.backend) {
    throw new Error("WebGPU backend not initialized, use loadBackend function");
  }
  return new DenseGPULayer(config, GPUInstance.backend);
};

const layers = {
  dense
}
const setup = async (silent = false) => {
  await GPUInstance.init(silent);
  Tensor.backend = new TensorGPUBackend(GPUInstance.backend!);
  Engine.backendLoader = loadBackend;
  Layer.layers = layers;
};



export const GPU = {
  loadBackend,
  model,
  setup,
  layers,
};

export { GPUBackend };
export * from "./matrix.ts";
