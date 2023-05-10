import { WASMBackend } from "./backend.ts";
import { NoBackendError } from "../core/api/error.ts";
import { BackendLoader, Engine } from "../core/engine.ts";
import { Backend, BackendType, NetworkConfig } from "../core/types.ts";
import { instantiate } from "./lib/netsaur.generated.js";

/**
 * Web Assembly backend instance.
 */
export class WASMInstance {
  static initialized = false;

  static async init(silent = false) {
    if (WASMInstance.initialized) return true;
    await instantiate();
    WASMInstance.initialized = true;
    if (!silent) console.log("WASM Backend Initialised");
    return true;
  }
}

/**
 * Web Assembly Backend Loader.
 */
export class WASMBackendLoader implements BackendLoader {
  async setup(silent = false) {
    Engine.type = BackendType.WASM;
    return await WASMInstance.init(silent);
  }

  loadBackend(config: NetworkConfig): Backend {
    if (!WASMInstance.initialized) {
      throw new NoBackendError(BackendType.WASM);
    }
    return new WASMBackend(config);
  }

  loadModel(path: string): Backend;
  loadModel(path: Uint8Array): Backend;
  loadModel(path: string | Uint8Array): Backend {
    return WASMBackend.loadModel(path);
  }
}

/**
 * Web Assembly Backend Type.
 */
export const WASM = new WASMBackendLoader();
