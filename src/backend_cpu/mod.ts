import { dlopen, FetchOptions } from "../../deps.ts";
import { CPUBackend } from "./backend.ts";
import { NoBackendError } from "../core/api/error.ts";
import { Engine } from "../core/engine.ts";
import { Tensor } from "../core/tensor/tensor.ts";
import { Backend, BackendType, NetworkConfig } from "../core/types.ts";
import { CPUTensorBackend } from "./tensor.ts";
import { NetworkJSON } from "../model/types.ts";

const options: FetchOptions = {
  name: "netsaur",
  url: new URL(import.meta.url).protocol !== "file:"
    ? new URL(
      "https://github.com/denosaurs/netsaur/releases/download/0.2.3/",
      import.meta.url,
    )
    : "./target/release/",
  cache: "reloadAll",
};

const symbols = {
  ffi_backend_create: {
    parameters: ["buffer", "usize"],
    result: "void",
  } as const,
  ffi_backend_train: {
    parameters: ["buffer", "usize", "buffer", "usize"],
    result: "void",
  } as const,
  ffi_backend_predict: {
    parameters: ["buffer", "buffer", "usize", "buffer"],
    result: "buffer",
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
    if (!silent) console.log("CPU Backend Initialised");
    return true;
  }
}

export class CPUBackendLoader {
  async setup(silent = false) {
    Tensor.backend = new CPUTensorBackend();
    Engine.type = BackendType.CPU;
    return await CPUInstance.init(silent);
  }

  loadBackend(config: NetworkConfig): Backend {
    if (!CPUInstance.initialized) {
      throw new NoBackendError(BackendType.CPU);
    }
    return new CPUBackend(config, CPUInstance.library!);
  }

  fromJSON(json: NetworkJSON): Backend {
    return CPUBackend.fromJSON(json) as Backend;
  }
}

/**
 * CPU Backend Type.
 */
export const CPU = new CPUBackendLoader();
