// deno-lint-ignore-file
// deno-fmt-ignore-file

export interface InstantiateResult {
  instance: WebAssembly.Instance;
  exports: {
    wasm_backend_create: typeof wasm_backend_create;
    wasm_backend_train: typeof wasm_backend_train;
    wasm_backend_predict: typeof wasm_backend_predict;
    wasm_backend_save: typeof wasm_backend_save;
    wasm_backend_load: typeof wasm_backend_load
  };
}

/** Gets if the Wasm module has been instantiated. */
export function isInstantiated(): boolean;

/** Options for instantiating a Wasm instance. */
export interface InstantiateOptions {
  /** Optional url to the Wasm file to instantiate. */
  url?: URL;
  /** Callback to decompress the raw Wasm file bytes before instantiating. */
  decompress?: (bytes: Uint8Array) => Uint8Array;
}

/** Instantiates an instance of the Wasm module returning its functions.
* @remarks It is safe to call this multiple times and once successfully
* loaded it will always return a reference to the same object. */
export function instantiate(opts?: InstantiateOptions): Promise<InstantiateResult["exports"]>;

/** Instantiates an instance of the Wasm module along with its exports.
 * @remarks It is safe to call this multiple times and once successfully
 * loaded it will always return a reference to the same object. */
export function instantiateWithInstance(opts?: InstantiateOptions): Promise<InstantiateResult>;

/**
* @param {string} config
* @param {Array<any>} shape
* @returns {number}
*/
export function wasm_backend_create(config: string, shape: Array<any>): number;
/**
* @param {number} id
* @param {(Float32Array)[]} buffers
* @param {string} options
*/
export function wasm_backend_train(id: number, buffers: (Float32Array)[], options: string): void;
/**
* @param {number} id
* @param {Float32Array} buffer
* @param {string} options
* @returns {Float32Array}
*/
export function wasm_backend_predict(id: number, buffer: Float32Array, options: string): Float32Array;
/**
* @param {number} id
* @returns {Uint8Array}
*/
export function wasm_backend_save(id: number): Uint8Array;
/**
* @param {Uint8Array} buffer
* @param {Array<any>} shape
* @returns {number}
*/
export function wasm_backend_load(buffer: Uint8Array, shape: Array<any>): number;
