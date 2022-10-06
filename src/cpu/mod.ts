import { Backend, NetworkConfig } from "../types.ts";

import { CPUBackend } from "./backend.ts";
// import { Backend, NetworkConfig } from "../types.ts";

// deno-lint-ignore require-await
export async function CPU(config: NetworkConfig): Promise<Backend> {
  return new CPUBackend(config);
}
