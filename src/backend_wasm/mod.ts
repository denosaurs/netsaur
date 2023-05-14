import { WASMBackend } from "./backend.ts";
import { NoBackendError } from "../core/api/error.ts";
import { BackendLoader, Engine } from "../core/engine.ts";
import { Backend, BackendType, Cost, NetworkConfig } from "../core/types.ts";
import { instantiate } from "./lib/netsaur.generated.js";
import { Sequential } from "../core/mod.ts";

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
  backend?: WASMBackend;

  async setup(silent = false) {
    Engine.type = BackendType.WASM;
    return await WASMInstance.init(silent);
  }

  loadBackend(config: NetworkConfig): Backend {
    if (!WASMInstance.initialized) {
      throw new NoBackendError(BackendType.WASM);
    }
    return this.backend ? this.backend : WASMBackend.create(config);
  }

  load(buffer: Uint8Array): Sequential {
    this.backend = WASMBackend.load(buffer);
    const net = new Sequential({ size: [0], layers: [], cost: Cost.MSE });
    this.backend = undefined;
    return net;
  }

  loadFile(path: string): Sequential {
    this.backend = WASMBackend.loadFile(path);
    const net = new Sequential({ size: [0], layers: [], cost: Cost.MSE });
    this.backend = undefined;
    return net;
  }
}

/**
 * Web Assembly Backend written in Rust & compiled to Web Assembly.
 */
export const WASM = new WASMBackendLoader();
