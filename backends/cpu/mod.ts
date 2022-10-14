import { Backend, NetworkConfig, NetworkJSON } from "../../core/types.ts";

import { CPUBackend } from "./backend.ts";
import { TensorCPUBackend } from "./tensor.ts";


export const CPU = {
  // deno-lint-ignore require-await
  backend: async (config: NetworkConfig): Promise<Backend> => new CPUBackend(config),
  // deno-lint-ignore require-await
  model: async (data: NetworkJSON, _silent=false): Promise<Backend> => CPUBackend.fromJSON(data),
  tensor: () => new TensorCPUBackend()
}
export { CPUBackend };
export * from "./matrix.ts";
export type { DataSet } from "../../core/types.ts";