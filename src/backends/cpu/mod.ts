import { dlopen, FetchOptions } from "../../../deps.ts";
import { CPUBackend } from "./backend.ts";
import { NoBackendError } from "../../core/api/error.ts";
import { BackendLoader, Engine } from "../../core/engine.ts";
import {
  Backend,
  BackendType,
  Cost,
  NetworkConfig,
  SchedulerType,
} from "../../core/types.ts";
import { Sequential } from "../../core/mod.ts";

const options: FetchOptions = {
  name: "netsaur",
  url: new URL(import.meta.url).protocol !== "file:"
    ? new URL(
      "https://github.com/denosaurs/netsaur/releases/download/0.2.14/",
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

export class CPUInstance {
  static library?: Library;
  static initialized = false;

  static async init(silent = false) {
    if (CPUInstance.initialized) return true;

    CPUInstance.library = await dlopen(options, symbols);
    CPUInstance.initialized = true;
    if (!silent) console.log("CPU Backend Initialized");
    return true;
  }
}

export class CPUBackendLoader implements BackendLoader {
  backend?: CPUBackend;

  isSupported(): boolean {
    return Deno.dlopen !== undefined;
  }

  async setup(silent = false) {
    Engine.type = BackendType.CPU;
    return await CPUInstance.init(silent);
  }

  loadBackend(config: NetworkConfig): Backend {
    if (!CPUInstance.initialized) {
      throw new NoBackendError(BackendType.CPU);
    }
    return this.backend
      ? this.backend
      : CPUBackend.create(config, CPUInstance.library!);
  }

  load(buffer: Uint8Array): Sequential {
    this.backend = CPUBackend.load(buffer, CPUInstance.library!);
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
    this.backend = CPUBackend.loadFile(path, CPUInstance.library!);
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
 * CPU Backend written in Rust.
 */
export const CPU = new CPUBackendLoader();
