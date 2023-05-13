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
    return WASMBackend.create(config);
  }

  load(data: Uint8Array): Backend {
    return WASMBackend.load(data);
  }

  loadFile(path: string): Backend {
    return WASMBackend.loadFile(path);
  }
}

/**
 * Web Assembly Backend written in Rust & compiled to Web Assembly.
 */
export const WASM = new WASMBackendLoader();
