// @generated file from wasmbuild -- do not edit
// deno-lint-ignore-file
// deno-fmt-ignore-file
// source-hash: 27a7ab3b4a6ba5708c4d244969e0c8d52811e2ac
let wasm;
let cachedInt32Memory0;

const heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

function getObject(idx) {
  return heap[idx];
}

let heap_next = heap.length;

function dropObject(idx) {
  if (idx < 132) return;
  heap[idx] = heap_next;
  heap_next = idx;
}

function takeObject(idx) {
  const ret = getObject(idx);
  dropObject(idx);
  return ret;
}

function addHeapObject(obj) {
  if (heap_next === heap.length) heap.push(heap.length + 1);
  const idx = heap_next;
  heap_next = heap[idx];

  heap[idx] = obj;
  return idx;
}

const cachedTextDecoder = typeof TextDecoder !== "undefined"
  ? new TextDecoder("utf-8", { ignoreBOM: true, fatal: true })
  : {
    decode: () => {
      throw Error("TextDecoder not available");
    },
  };

if (typeof TextDecoder !== "undefined") cachedTextDecoder.decode();

let cachedUint8Memory0 = null;

function getUint8Memory0() {
  if (cachedUint8Memory0 === null || cachedUint8Memory0.byteLength === 0) {
    cachedUint8Memory0 = new Uint8Array(wasm.memory.buffer);
  }
  return cachedUint8Memory0;
}

function getStringFromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return cachedTextDecoder.decode(getUint8Memory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = typeof TextEncoder !== "undefined"
  ? new TextEncoder("utf-8")
  : {
    encode: () => {
      throw Error("TextEncoder not available");
    },
  };

const encodeString = function (arg, view) {
  return cachedTextEncoder.encodeInto(arg, view);
};

function passStringToWasm0(arg, malloc, realloc) {
  if (realloc === undefined) {
    const buf = cachedTextEncoder.encode(arg);
    const ptr = malloc(buf.length, 1) >>> 0;
    getUint8Memory0().subarray(ptr, ptr + buf.length).set(buf);
    WASM_VECTOR_LEN = buf.length;
    return ptr;
  }

  let len = arg.length;
  let ptr = malloc(len, 1) >>> 0;

  const mem = getUint8Memory0();

  let offset = 0;

  for (; offset < len; offset++) {
    const code = arg.charCodeAt(offset);
    if (code > 0x7F) break;
    mem[ptr + offset] = code;
  }

  if (offset !== len) {
    if (offset !== 0) {
      arg = arg.slice(offset);
    }
    ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
    const view = getUint8Memory0().subarray(ptr + offset, ptr + len);
    const ret = encodeString(arg, view);

    offset += ret.written;
  }

  WASM_VECTOR_LEN = offset;
  return ptr;
}
/**
 * @param {string} config
 * @param {Array<any>} shape
 * @returns {number}
 */
export function wasm_backend_create(config, shape) {
  const ptr0 = passStringToWasm0(
    config,
    wasm.__wbindgen_malloc,
    wasm.__wbindgen_realloc,
  );
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_backend_create(ptr0, len0, addHeapObject(shape));
  return ret >>> 0;
}

let cachedUint32Memory0 = null;

function getUint32Memory0() {
  if (cachedUint32Memory0 === null || cachedUint32Memory0.byteLength === 0) {
    cachedUint32Memory0 = new Uint32Array(wasm.memory.buffer);
  }
  return cachedUint32Memory0;
}

function passArrayJsValueToWasm0(array, malloc) {
  const ptr = malloc(array.length * 4, 4) >>> 0;
  const mem = getUint32Memory0();
  for (let i = 0; i < array.length; i++) {
    mem[ptr / 4 + i] = addHeapObject(array[i]);
  }
  WASM_VECTOR_LEN = array.length;
  return ptr;
}
/**
 * @param {number} id
 * @param {(Float32Array)[]} buffers
 * @param {string} options
 */
export function wasm_backend_train(id, buffers, options) {
  const ptr0 = passArrayJsValueToWasm0(buffers, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ptr1 = passStringToWasm0(
    options,
    wasm.__wbindgen_malloc,
    wasm.__wbindgen_realloc,
  );
  const len1 = WASM_VECTOR_LEN;
  wasm.wasm_backend_train(id, ptr0, len0, ptr1, len1);
}

/**
 * @param {number} id
 * @param {Float32Array} buffer
 * @param {string} options
 * @returns {Float32Array}
 */
export function wasm_backend_predict(id, buffer, options) {
  const ptr0 = passStringToWasm0(
    options,
    wasm.__wbindgen_malloc,
    wasm.__wbindgen_realloc,
  );
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_backend_predict(id, addHeapObject(buffer), ptr0, len0);
  return takeObject(ret);
}

/**
 * @param {number} id
 * @returns {Uint8Array}
 */
export function wasm_backend_save(id) {
  const ret = wasm.wasm_backend_save(id);
  return takeObject(ret);
}

/**
 * @param {Uint8Array} buffer
 * @param {Array<any>} shape
 * @returns {number}
 */
export function wasm_backend_load(buffer, shape) {
  const ret = wasm.wasm_backend_load(
    addHeapObject(buffer),
    addHeapObject(shape),
  );
  return ret >>> 0;
}

function handleError(f, args) {
  try {
    return f.apply(this, args);
  } catch (e) {
    wasm.__wbindgen_exn_store(addHeapObject(e));
  }
}

const imports = {
  __wbindgen_placeholder__: {
    __wbindgen_object_drop_ref: function (arg0) {
      takeObject(arg0);
    },
    __wbg_log_8d8708b5d26d7ac1: function (arg0, arg1) {
      console.log(getStringFromWasm0(arg0, arg1));
    },
    __wbindgen_number_new: function (arg0) {
      const ret = arg0;
      return addHeapObject(ret);
    },
    __wbg_crypto_c48a774b022d20ac: function (arg0) {
      const ret = getObject(arg0).crypto;
      return addHeapObject(ret);
    },
    __wbindgen_is_object: function (arg0) {
      const val = getObject(arg0);
      const ret = typeof val === "object" && val !== null;
      return ret;
    },
    __wbg_process_298734cf255a885d: function (arg0) {
      const ret = getObject(arg0).process;
      return addHeapObject(ret);
    },
    __wbg_versions_e2e78e134e3e5d01: function (arg0) {
      const ret = getObject(arg0).versions;
      return addHeapObject(ret);
    },
    __wbg_node_1cd7a5d853dbea79: function (arg0) {
      const ret = getObject(arg0).node;
      return addHeapObject(ret);
    },
    __wbindgen_is_string: function (arg0) {
      const ret = typeof (getObject(arg0)) === "string";
      return ret;
    },
    __wbg_msCrypto_bcb970640f50a1e8: function (arg0) {
      const ret = getObject(arg0).msCrypto;
      return addHeapObject(ret);
    },
    __wbg_require_8f08ceecec0f4fee: function () {
      return handleError(function () {
        const ret = module.require;
        return addHeapObject(ret);
      }, arguments);
    },
    __wbindgen_is_function: function (arg0) {
      const ret = typeof (getObject(arg0)) === "function";
      return ret;
    },
    __wbindgen_string_new: function (arg0, arg1) {
      const ret = getStringFromWasm0(arg0, arg1);
      return addHeapObject(ret);
    },
    __wbg_getRandomValues_37fa2ca9e4e07fab: function () {
      return handleError(function (arg0, arg1) {
        getObject(arg0).getRandomValues(getObject(arg1));
      }, arguments);
    },
    __wbg_randomFillSync_dc1e9a60c158336d: function () {
      return handleError(function (arg0, arg1) {
        getObject(arg0).randomFillSync(takeObject(arg1));
      }, arguments);
    },
    __wbg_newnoargs_581967eacc0e2604: function (arg0, arg1) {
      const ret = new Function(getStringFromWasm0(arg0, arg1));
      return addHeapObject(ret);
    },
    __wbg_call_cb65541d95d71282: function () {
      return handleError(function (arg0, arg1) {
        const ret = getObject(arg0).call(getObject(arg1));
        return addHeapObject(ret);
      }, arguments);
    },
    __wbindgen_object_clone_ref: function (arg0) {
      const ret = getObject(arg0);
      return addHeapObject(ret);
    },
    __wbg_self_1ff1d729e9aae938: function () {
      return handleError(function () {
        const ret = self.self;
        return addHeapObject(ret);
      }, arguments);
    },
    __wbg_window_5f4faef6c12b79ec: function () {
      return handleError(function () {
        const ret = window.window;
        return addHeapObject(ret);
      }, arguments);
    },
    __wbg_globalThis_1d39714405582d3c: function () {
      return handleError(function () {
        const ret = globalThis.globalThis;
        return addHeapObject(ret);
      }, arguments);
    },
    __wbg_global_651f05c6a0944d1c: function () {
      return handleError(function () {
        const ret = global.global;
        return addHeapObject(ret);
      }, arguments);
    },
    __wbindgen_is_undefined: function (arg0) {
      const ret = getObject(arg0) === undefined;
      return ret;
    },
    __wbg_set_502d29070ea18557: function (arg0, arg1, arg2) {
      getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
    },
    __wbg_setlength_02cc4ef174b232f4: function (arg0, arg1) {
      getObject(arg0).length = arg1 >>> 0;
    },
    __wbg_call_01734de55d61e11d: function () {
      return handleError(function (arg0, arg1, arg2) {
        const ret = getObject(arg0).call(getObject(arg1), getObject(arg2));
        return addHeapObject(ret);
      }, arguments);
    },
    __wbg_buffer_085ec1f694018c4f: function (arg0) {
      const ret = getObject(arg0).buffer;
      return addHeapObject(ret);
    },
    __wbg_newwithbyteoffsetandlength_6da8e527659b86aa: function (
      arg0,
      arg1,
      arg2,
    ) {
      const ret = new Uint8Array(getObject(arg0), arg1 >>> 0, arg2 >>> 0);
      return addHeapObject(ret);
    },
    __wbg_new_8125e318e6245eed: function (arg0) {
      const ret = new Uint8Array(getObject(arg0));
      return addHeapObject(ret);
    },
    __wbg_set_5cf90238115182c3: function (arg0, arg1, arg2) {
      getObject(arg0).set(getObject(arg1), arg2 >>> 0);
    },
    __wbg_length_72e2208bbc0efc61: function (arg0) {
      const ret = getObject(arg0).length;
      return ret;
    },
    __wbg_newwithbyteoffsetandlength_69193e31c844b792: function (
      arg0,
      arg1,
      arg2,
    ) {
      const ret = new Float32Array(getObject(arg0), arg1 >>> 0, arg2 >>> 0);
      return addHeapObject(ret);
    },
    __wbg_new_d086a66d1c264b3f: function (arg0) {
      const ret = new Float32Array(getObject(arg0));
      return addHeapObject(ret);
    },
    __wbg_set_6146c51d49a2c0df: function (arg0, arg1, arg2) {
      getObject(arg0).set(getObject(arg1), arg2 >>> 0);
    },
    __wbg_length_d7327c75a759af37: function (arg0) {
      const ret = getObject(arg0).length;
      return ret;
    },
    __wbg_newwithlength_e5d69174d6984cd7: function (arg0) {
      const ret = new Uint8Array(arg0 >>> 0);
      return addHeapObject(ret);
    },
    __wbg_subarray_13db269f57aa838d: function (arg0, arg1, arg2) {
      const ret = getObject(arg0).subarray(arg1 >>> 0, arg2 >>> 0);
      return addHeapObject(ret);
    },
    __wbindgen_throw: function (arg0, arg1) {
      throw new Error(getStringFromWasm0(arg0, arg1));
    },
    __wbindgen_memory: function () {
      const ret = wasm.memory;
      return addHeapObject(ret);
    },
  },
};

import { Loader } from "https://deno.land/x/wasmbuild@0.15.1/loader.ts";
import { cacheToLocalDir } from "https://deno.land/x/wasmbuild@0.15.1/cache.ts";

const loader = new Loader({
  imports,
  cache: cacheToLocalDir,
});
/**
 * Decompression callback
 *
 * @callback DecompressCallback
 * @param {Uint8Array} compressed
 * @return {Uint8Array} decompressed
 */

/**
 * Options for instantiating a Wasm instance.
 * @typedef {Object} InstantiateOptions
 * @property {URL=} url - Optional url to the Wasm file to instantiate.
 * @property {DecompressCallback=} decompress - Callback to decompress the
 * raw Wasm file bytes before instantiating.
 */

/** Instantiates an instance of the Wasm module returning its functions.
 * @remarks It is safe to call this multiple times and once successfully
 * loaded it will always return a reference to the same object.
 * @param {InstantiateOptions=} opts
 */
export async function instantiate(opts) {
  return (await instantiateWithInstance(opts)).exports;
}

/** Instantiates an instance of the Wasm module along with its exports.
 * @remarks It is safe to call this multiple times and once successfully
 * loaded it will always return a reference to the same object.
 * @param {InstantiateOptions=} opts
 * @returns {Promise<{
 *   instance: WebAssembly.Instance;
 *   exports: { wasm_backend_create: typeof wasm_backend_create; wasm_backend_train: typeof wasm_backend_train; wasm_backend_predict: typeof wasm_backend_predict; wasm_backend_save: typeof wasm_backend_save; wasm_backend_load: typeof wasm_backend_load }
 * }>}
 */
export async function instantiateWithInstance(opts) {
  const { instance } = await loader.load(
    opts?.url ?? new URL("netsaur_bg.wasm", import.meta.url),
    opts?.decompress,
  );
  wasm = wasm ?? instance.exports;
  cachedInt32Memory0 = cachedInt32Memory0 ?? new Int32Array(wasm.memory.buffer);
  cachedUint8Memory0 = cachedUint8Memory0 ?? new Uint8Array(wasm.memory.buffer);
  return {
    instance,
    exports: getWasmInstanceExports(),
  };
}

function getWasmInstanceExports() {
  return {
    wasm_backend_create,
    wasm_backend_train,
    wasm_backend_predict,
    wasm_backend_save,
    wasm_backend_load,
  };
}

/** Gets if the Wasm module has been instantiated. */
export function isInstantiated() {
  return loader.instance != null;
}
