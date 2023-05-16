export { setupBackend } from "./src/core/engine.ts";
export * from "./src/core/mod.ts";
export * from "./src/core/types.ts";
export * from "./src/core/tensor/tensor.ts";
export * from "./src/core/api/layers.ts";
export * from "./src/core/api/shape.ts";
export * from "./src/core/api/network.ts";

import { CPU } from "./src/backend_cpu/mod.ts";
import { WASM } from "./src/backend_wasm/mod.ts";

/**
 * The AUTO backend is chosen automatically based on the environment.
 */
const AUTO = Deno.dlopen === undefined ? WASM : CPU;

export { AUTO, CPU, WASM };
