// deno-lint-ignore-file
// deno-fmt-ignore-file

export interface InstantiateResult {
  instance: WebAssembly.Instance;
  exports: {
    wasm_tokenizer_from_json: typeof wasm_tokenizer_from_json;
    wasm_tokenizer_save: typeof wasm_tokenizer_save;
    wasm_bpe_default: typeof wasm_bpe_default;
    wasm_tokenizer_encode: typeof wasm_tokenizer_encode;
    wasm_tokenizer_get_vocab: typeof wasm_tokenizer_get_vocab;
    wasm_tokenizer_get_vocab_size: typeof wasm_tokenizer_get_vocab_size;
    wasm_tokenizer_decode: typeof wasm_tokenizer_decode;
    wasm_tokenizer_token_to_id: typeof wasm_tokenizer_token_to_id;
    wasm_tokenizer_id_to_token: typeof wasm_tokenizer_id_to_token
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
* @param {string} json
* @returns {number}
*/
export function wasm_tokenizer_from_json(json: string): number;
/**
* @param {number} id
* @param {boolean} pretty
* @returns {string}
*/
export function wasm_tokenizer_save(id: number, pretty: boolean): string;
/**
* @returns {number}
*/
export function wasm_bpe_default(): number;
/**
* @param {number} id
* @param {string} string
* @returns {Uint32Array}
*/
export function wasm_tokenizer_encode(id: number, string: string): Uint32Array;
/**
* @param {number} id
* @param {boolean} with_added_tokens
* @returns {any}
*/
export function wasm_tokenizer_get_vocab(id: number, with_added_tokens: boolean): any;
/**
* @param {number} id
* @param {boolean} with_added_tokens
* @returns {number}
*/
export function wasm_tokenizer_get_vocab_size(id: number, with_added_tokens: boolean): number;
/**
* @param {number} id
* @param {Uint32Array} ids
* @param {boolean} skip_special_tokens
* @returns {string}
*/
export function wasm_tokenizer_decode(id: number, ids: Uint32Array, skip_special_tokens: boolean): string;
/**
* @param {number} id
* @param {string} token
* @returns {number}
*/
export function wasm_tokenizer_token_to_id(id: number, token: string): number;
/**
* @param {number} id
* @param {number} token_id
* @returns {string}
*/
export function wasm_tokenizer_id_to_token(id: number, token_id: number): string;
