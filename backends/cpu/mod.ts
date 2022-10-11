import { Backend, NetworkConfig, NetworkJSON } from "../../core/types.ts";

import { CPUBackend } from "./backend.ts";

// deno-lint-ignore require-await
export async function CPU(config: NetworkConfig): Promise<Backend> {
  return new CPUBackend(config);
}

// deno-lint-ignore require-await
export async function CPUModel(data: NetworkJSON, _silent=false): Promise<Backend> {
  return CPUBackend.fromJSON(data);
}

export { CPUBackend };
export * from "./matrix.ts";
export type { DataSet } from "../../core/types.ts";