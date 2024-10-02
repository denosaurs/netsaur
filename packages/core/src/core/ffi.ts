import { dlopen, type FetchOptions } from "../../../../deps.ts";

const options: FetchOptions = {
  name: "netsaur_modular",
  url:
    new URL(import.meta.url).protocol !== "file:"
      ? new URL(
          "https://github.com/denosaurs/netsaur/releases/download/0.4.2/",
          import.meta.url
        )
      : "./target/release/",
  cache: "reloadAll",
};

const symbols = {
  ffi_layer_create: {
    parameters: ["buffer", "usize", "buffer", "usize"],
    result: "usize",
  } as const,
  ffi_layer_forward: {
    parameters: ["usize", "buffer", "buffer", "usize", "buffer", "bool"],
    result: "void",
  } as const,
  ffi_layer_backward: {
    parameters: ["usize", "buffer", "buffer", "usize", "buffer"],
    result: "void",
  } as const,
  ffi_optimizer_create: {
    parameters: ["buffer", "usize", "buffer", "usize"],
    result: "usize",
  } as const,
  ffi_optimize: {
    parameters: ["usize", "f32", "usize", "buffer", "usize"],
    result: "void",
  } as const,
  ffi_cost_create: {
    parameters: ["buffer", "usize"],
    result: "usize",
  } as const,
  ffi_cost: {
    parameters: ["usize", "buffer", "usize", "buffer", "buffer"],
    result: "f32",
  } as const,
  ffi_cost_d: {
    parameters: ["usize", "buffer", "usize", "buffer", "buffer", "buffer"],
    result: "void",
  } as const,
};

type Library = Deno.DynamicLibrary<typeof symbols>;

class CPUInstance {
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

await CPUInstance.init();

export default CPUInstance.library as Library;