import { dlopen, FetchOptions } from "../../../deps.ts";
import { GPUBackend } from "./backend.ts";
import { NoBackendError } from "../../core/api/error.ts";
import { BackendLoader, Engine } from "../../core/engine.ts";
import { Backend, BackendType, Cost, NetworkConfig } from "../../core/types.ts";
import { Sequential } from "../../core/mod.ts";

const options: FetchOptions = {
  name: "netsaur_gpu",
  url: new URL(import.meta.url).protocol !== "file:"
    ? new URL(
      "https://github.com/denosaurs/netsaur/releases/download/0.2.10/",
      import.meta.url,
    )
    : "./target/release/",
  cache: "reloadAll",
};

const symbols = {
  ffi_backend_create: {
    parameters: ["buffer", "usize", "pointer"],
    result: "usize",
  } as const,
  ffi_backend_train: {
    parameters: ["usize", "buffer", "usize", "buffer", "usize"],
    result: "void",
  } as const,
  ffi_backend_predict: {
    parameters: ["usize", "buffer", "buffer", "usize", "buffer"],
    result: "void",
  } as const,
  ffi_backend_save: {
    parameters: ["usize", "pointer"],
    result: "void",
  } as const,
  ffi_backend_load: {
    parameters: ["buffer", "usize", "pointer"],
    result: "usize",
  } as const,
};

export type Library = Deno.DynamicLibrary<typeof symbols>;

export class GPUInstance {
  static library?: Library;
  static initialized = false;

  static async init(silent = false) {
    if (GPUInstance.initialized) return true;

    GPUInstance.library = await dlopen(options, symbols);
    GPUInstance.initialized = true;
    if (!silent) console.log("GPU Backend Initialized");
    return true;
  }
}

export class GPUBackendLoader implements BackendLoader {
  backend?: GPUBackend;

  isSupported(): boolean {
    return Deno.dlopen !== undefined;
  }

  async setup(silent = false) {
    Engine.type = BackendType.GPU;
    return await GPUInstance.init(silent);
  }

  loadBackend(config: NetworkConfig): Backend {
    if (!GPUInstance.initialized) {
      throw new NoBackendError(BackendType.GPU);
    }
    return this.backend
      ? this.backend
      : GPUBackend.create(config, GPUInstance.library!);
  }

  load(buffer: Uint8Array): Sequential {
    this.backend = GPUBackend.load(buffer, GPUInstance.library!);
    const net = new Sequential({ size: [0], layers: [], cost: Cost.MSE });
    this.backend = undefined;
    return net;
  }

  loadFile(path: string): Sequential {
    this.backend = GPUBackend.loadFile(path, GPUInstance.library!);
    const net = new Sequential({ size: [0], layers: [], cost: Cost.MSE });
    this.backend = undefined;
    return net;
  }
}

/**
 * GPU Backend written in Rust.
 */
export const GPU = new GPUBackendLoader();
