import { Backend, NetworkConfig } from "../../core/types.ts";

import { NativeBackend } from "./backend.ts";
import { TensorNativeBackend } from "./tensor.ts";

export const Native = {
  // deno-lint-ignore require-await
  backend: async (config: NetworkConfig): Promise<Backend> =>
    // deno-lint-ignore no-explicit-any
    new NativeBackend(config as any),
  tensor: () => new TensorNativeBackend(),
};
export * from "./backend.ts";
export * from "./matrix.ts";
export type { DataSet } from "../../core/types.ts";
