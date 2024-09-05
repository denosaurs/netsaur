export { setupBackend } from "./packages/core/core/engine.ts";
export * from "./packages/core/core/mod.ts";
export * from "./packages/core/core/types.ts";
export * from "./packages/core/core/tensor/tensor.ts";
export * from "./packages/core/core/api/layers.ts";
export * from "./packages/core/core/api/shape.ts";
export * from "./packages/core/core/api/network.ts";
export * from "./packages/core/core/api/optimizer.ts";
export * from "./packages/core/core/api/scheduler.ts";
export { GPU } from "./packages/core/backends/gpu/mod.ts";

import { CPU, type CPUBackendLoader } from "./packages/core/backends/cpu/mod.ts";
import { WASM, type WASMBackendLoader } from "./packages/core/backends/wasm/mod.ts";
import type { BackendLoader } from "./packages/core/core/engine.ts";

onerror = () => {
  if (typeof Deno == "undefined") {
    throw new Error(
      "Warning: Deno is not defined. Did you mean to import from ./web.ts instead of ./mod.ts?",
    );
  }
};

/**
 * The AUTO backend is chosen automatically based on the environment.
 */
const AUTO: WASMBackendLoader | CPUBackendLoader = Deno.dlopen === undefined
  ? WASM
  : CPU;

/**
 * The OPTION function is used to choose a backend from a list of options.
 */
export function OPTION(...backends: BackendLoader[]): BackendLoader {
  for (const backend of backends) {
    if (backend.isSupported()) {
      return backend;
    }
  }
  throw new Error("No provided backend is supported");
}
export { AUTO, CPU, WASM };
