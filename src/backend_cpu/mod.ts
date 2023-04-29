import { dlopen, FetchOptions } from "../../deps.ts";
import { CPUBackend } from "./backend.ts";
import { NoDynamicLibraryError } from "../core/api/error.ts";
import { Engine } from "../core/engine.ts";
import { Tensor } from "../core/tensor/tensor.ts";
import { Backend, BackendType, NetworkConfig } from "../core/types.ts";
import { CPUTensorBackend } from "./tensor.ts";

const options: FetchOptions = {
  name: "netsaur",
  // url: `${import.meta.url}/../../release/`,
  url: "./target/debug/",
  cache: "reloadAll",
};

const symbols = {
  ops_backend_create: {
    parameters: ["buffer", "usize"],
    result: "void",
  } as const,
};

export class CPUInstance {
  static library?: Deno.DynamicLibrary<typeof symbols>;
  static initialized = false;

  static async init(silent = false) {
    if (CPUInstance.initialized) return;

    CPUInstance.library = await dlopen(options, symbols);
    CPUInstance.initialized = true;
    if (!silent) console.log("CPU Backend Initialised");
    return true;
  }
}

async function setup(silent = false) {
  await CPUInstance.init(silent);
  Tensor.backend = new CPUTensorBackend();
  Engine.type = BackendType.CPU;
}

function loadBackend(config: NetworkConfig): Backend {
  if (!CPUInstance.initialized) throw new NoDynamicLibraryError();
  return new CPUBackend(config, CPUInstance.library!);
}

export const CPU = {
  setup,
  loadBackend,
};


// Test

const config = JSON.stringify({
  size: [0],
  layers: [
    {
      type: "dense",
      config: {
        size: [0],
      },
    },
  ],
  cost: "mse",
});

const buffer = new TextEncoder().encode(config);

CPUInstance.library!.symbols.ops_backend_create(buffer, buffer.length);
