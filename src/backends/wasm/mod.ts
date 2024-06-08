import { WASMBackend } from "./backend.ts";
import { NoBackendError } from "../../core/api/error.ts";
import { type BackendLoader, Engine } from "../../core/engine.ts";
import {
  type Backend,
  BackendType,
  Cost,
  type NetworkConfig,
  SchedulerType,
} from "../../core/types.ts";
import { instantiate } from "./lib/netsaur.generated.js";
import { Sequential } from "../../core/mod.ts";

/**
 * Web Assembly backend instance.
 */
export class WASMInstance {
  static initialized = false;

  static async init(silent = false): Promise<boolean> {
    if (WASMInstance.initialized) return true;
    await instantiate({
      url: new URL(import.meta.url).protocol !== "file:"
        ? new URL(
          "https://github.com/denosaurs/netsaur/releases/download/0.3.2/netsaur_bg.wasm",
          import.meta.url,
        )
        : undefined,
    });
    WASMInstance.initialized = true;
    if (!silent) console.log("WASM Backend Initialized");
    return true;
  }
}

/**
 * Web Assembly Backend Loader.
 */
export class WASMBackendLoader implements BackendLoader {
  backend?: WASMBackend;

  isSupported(): boolean {
    return true;
  }

  async setup(silent = false): Promise<boolean> {
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
    const net = new Sequential({
      size: [0],
      layers: [],
      cost: Cost.MSE,
      scheduler: {
        type: SchedulerType.None,
      },
    });
    this.backend = undefined;
    return net;
  }

  loadFile(path: string): Sequential {
    this.backend = WASMBackend.loadFile(path);
    const net = new Sequential({
      size: [0],
      layers: [],
      cost: Cost.MSE,
      scheduler: {
        type: SchedulerType.None,
      },
    });
    this.backend = undefined;
    return net;
  }
}

/**
 * Web Assembly Backend written in Rust & compiled to Web Assembly.
 */
export const WASM: WASMBackendLoader = new WASMBackendLoader();
