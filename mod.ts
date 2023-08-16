export { setupBackend } from "./src/core/engine.ts";
export * from "./src/core/mod.ts";
export * from "./src/core/types.ts";
export * from "./src/core/tensor/tensor.ts";
export * from "./src/core/api/layers.ts";
export * from "./src/core/api/shape.ts";
export * from "./src/core/api/network.ts";

import { CPU } from "./src/backends/cpu/mod.ts";
import { WASM } from "./src/backends/wasm/mod.ts";
import { BackendLoader } from "./src/core/engine.ts";

/**
 * The AUTO backend is chosen automatically based on the environment.
 */
const AUTO = Deno.dlopen === undefined ? WASM : CPU;

/**
 * The OPTION function is used to choose a backend from a list of options.
 */
export function OPTION(...backends: BackendLoader[]) {
  for (const backend of backends) {
    if (backend.isSupported()) {
      return backend;
    }
  }
  throw new Error("No provided backend is supported");
}
export { AUTO, CPU, WASM };
