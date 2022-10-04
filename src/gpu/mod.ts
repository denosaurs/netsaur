import { Backend, NetworkConfig } from "../types.ts";

import { Core, WebGPUBackend } from "../../deps.ts";
import { GPUBackend } from "./backend.ts";

export async function GPU(config: NetworkConfig): Promise<Backend> {
  const silent = config.silent;
  const core = new Core();
  await core.initialize();
  const backend = core.backends.get("webgpu")! as WebGPUBackend;
  if (!backend.adapter) throw new Error("No backend adapter found!");
  if (!silent) console.log(`Using adapter: ${backend.adapter}`);
  const features = [...backend.adapter.features.values()];
  if (!silent) console.log(`Supported features: ${features.join(", ")}`);
  return new GPUBackend(config, backend);
}
