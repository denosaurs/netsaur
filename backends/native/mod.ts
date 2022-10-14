import { Backend, NetworkConfig } from "../../core/types.ts";

import { NativeBackend } from "./backend.ts";

export const Native = {
// deno-lint-ignore require-await no-explicit-any
  backend: async (config: NetworkConfig): Promise<Backend> => new NativeBackend(config as any),
}
export * from "./backend.ts";
export * from "./matrix.ts";
export type { DataSet } from "../../core/types.ts";
