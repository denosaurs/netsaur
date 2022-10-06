import { prepare } from "https://deno.land/x/plug@0.5.2/plug.ts"

const symbols = {
  matrix_new: {
    parameters: ["i32", "i32", "u8"],
    result: "pointer",
  },

  matrix_new_randf: {
    parameters: ["i32", "i32"],
    result: "pointer",
  },

  matrix_new_from_array: {
    parameters: ["i32", "i32", "u8", "buffer"],
    result: "pointer",
  },

  matrix_new_fill_u32: {
    parameters: ["i32", "i32", "u32"],
    result: "pointer",
  },

  matrix_new_fill_f32: {
    parameters: ["i32", "i32", "f32"],
    result: "pointer",
  },

  matrix_new_fill_i32: {
    parameters: ["i32", "i32", "i32"],
    result: "pointer",
  },

  matrix_copy: {
    parameters: ["pointer"],
    result: "pointer",
  },

  matrix_dot: {
    parameters: ["pointer", "pointer"],
    result: "pointer",
  },

  matrix_add: {
    parameters: ["pointer", "pointer"],
    result: "pointer",
  },

  matrix_sub: {
    parameters: ["pointer", "pointer"],
    result: "pointer",
  },

  matrix_add_f32: {
    parameters: ["pointer", "f32"],
    result: "pointer",
  },

  matrix_sub_f32: {
    parameters: ["pointer", "f32"],
    result: "pointer",
  },

  matrix_mul_f32: {
    parameters: ["pointer", "f32"],
    result: "pointer",
  },

  matrix_div_f32: {
    parameters: ["pointer", "f32"],
    result: "pointer",
  },

  matrix_add_i32: {
    parameters: ["pointer", "i32"],
    result: "pointer",
  },

  matrix_sub_i32: {
    parameters: ["pointer", "i32"],
    result: "pointer",
  },

  matrix_mul_i32: {
    parameters: ["pointer", "i32"],
    result: "pointer",
  },

  matrix_div_i32: {
    parameters: ["pointer", "i32"],
    result: "pointer",
  },

  matrix_add_u32: {
    parameters: ["pointer", "u32"],
    result: "pointer",
  },

  matrix_sub_u32: {
    parameters: ["pointer", "u32"],
    result: "pointer",
  },

  matrix_mul_u32: {
    parameters: ["pointer", "u32"],
    result: "pointer",
  },

  matrix_div_u32: {
    parameters: ["pointer", "u32"],
    result: "pointer",
  },

  matrix_transpose: {
    parameters: ["pointer"],
    result: "pointer",
  },

  matrix_print: {
    parameters: ["pointer", "buffer"],
    result: "void",
  },

  matrix_free: {
    parameters: ["pointer"],
    result: "void",
  },

  layer_dense: {
    parameters: ["u32", "u8"],
    result: "pointer",
  },

  layer_free: {
    parameters: ["pointer"],
    result: "void",
  },

  network_create: {
    parameters: ["u32", "u8", "u32", "buffer"],
    result: "pointer",
  },

  network_feed_forward: {
    parameters: ["pointer", "pointer"],
    result: "pointer",
  },

  network_train: {
    parameters: ["pointer", "u32", "buffer", "u32", "f32"],
    result: "void",
  },

  network_free: {
    parameters: ["pointer"],
    result: "void",
  },

  network_save: {
    parameters: ["pointer", "buffer"],
    result: "void",
  },

  network_load: {
    parameters: ["buffer"],
    result: "pointer",
  },
} as const;

const url = new URL(
  "https://github.com/denosaurs/netsaur/releases/download/0.1.3/",
  import.meta.url,
)
let uri = url.toString()
if (!uri.endsWith("/")) uri += "/"

let darwin: string | { aarch64: string; x86_64: string } = uri + "libnetsaur.dylib"

if (url.protocol !== "file:") {
  darwin = {
    aarch64: uri + "libnetsaur_arm64.dylib",
    x86_64: uri + "libnetsaur.dylib",
  }
}

const opts = {
  name: "netsaur",
  urls: {
    darwin,
    windows: uri + "libnetsaur.dll",
    linux: uri + "libnetsaur.so",
  },
  policy: undefined,
}


const mod = await prepare(opts, symbols);
export default mod.symbols;

export function cstr(str: string) {
  return new TextEncoder().encode(str + "\0");
}
