import { WASM } from "../backends/wasm/mod.ts";
import type { Sequential } from "./mod.ts";
import type { Backend, BackendType, NetworkConfig } from "./types.ts";

export interface BackendInstance {
  /**
   * Initialize the backend.
   */
  init(): Promise<void>;
}

/**
 * The Backend Loader is responsible for loading the backend and setting it up.
 */
export interface BackendLoader {
  /**
   * Whether the backend is supported.
   * 
   * ```ts
   * import { WASM } from "https://deno.land/x/netsaur/mod.ts";
   * 
   * console.log(WASM.isSupported());
   * ```
   */
  isSupported(): boolean;

  /**
   * Setup the backend. Returns true if the backend was successfully setup.
   */
  setup(silent: boolean): Promise<boolean>;

  /**
   * Load the backend from a config.
   */
  loadBackend(config: NetworkConfig): Backend;

  /**
   * Load a model from a safe tensors file path.
   */
  loadFile(path: string): Sequential;

  /**
   * Load a model from Uint8Array data.
   */
  load(data: Uint8Array): Sequential;
}

/**
 * setupBackend loads the backend and sets it up.
 * ```ts
 * import { setupBackend, CPU } from "https://deno.land/x/netsaur/mod.ts";
 *
 * await setupBackend(CPU);
 * ```
 */
export async function setupBackend(
  loader: BackendLoader,
  silent = false,
): Promise<void> {
  const success = await loader.setup(silent);
  if (!success) {
    if (!silent) console.log("Defaulting to CPU Backend");
    await WASM.setup(silent);
  }
  Engine.backendLoader = loader;
}

/**
 * the Engine manages the backend and the backend loader.
 */
export class Engine {
  static backendLoader: BackendLoader;

  static type: BackendType;
}
