import { Backend, NetworkConfig, NetworkJSON } from "../types.ts";

import { CPUBackend } from "./backend.ts";
// import { Backend, NetworkConfig } from "../types.ts";

// deno-lint-ignore require-await
export async function CPU(config: NetworkConfig): Promise<Backend> {
  return new CPUBackend(config);
}

// deno-lint-ignore require-await
export async function CPUModel(data: NetworkJSON, _silent=false): Promise<Backend> {
  return CPUBackend.fromJSON(data);
}