import { CPU } from "../backend_cpu/mod.ts";
import { Backend, BackendType, NetworkConfig } from "./types.ts";

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
  loadFile(path: string): Backend;

  /**
   * Load a model from Uint8Array data.
   */
  load(data: Uint8Array): Backend;
}

/**
 * setupBackend loads the backend and sets it up.
 */
export async function setupBackend(loader: BackendLoader, silent = false) {
  const success = await loader.setup(silent);
  if (!success) {
    if (!silent) console.log("Defaulting to CPU Backend");
    await CPU.setup(silent);
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
