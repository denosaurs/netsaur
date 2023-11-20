import {
  instantiate,
  wasm_tokenizer_decode,
  wasm_tokenizer_encode,
  wasm_tokenizer_from_json,
  wasm_tokenizer_get_vocab,
  wasm_tokenizer_get_vocab_size,
  wasm_tokenizer_save,
  wasm_tokenizer_id_to_token,
  wasm_tokenizer_token_to_id,
} from "./lib/netsaur_tokenizers.generated.js";

let initialized = false;
export async function init() {
  if (initialized) return;
  await instantiate({
    url: new URL(import.meta.url).protocol !== "file:"
      ? new URL(
        "https://github.com/denosaurs/netsaur/releases/download/0.2.15/netsaur_tokenizers_bg.wasm",
        import.meta.url,
      )
      : undefined,
  });
  initialized = true;
}

/**
 * Tokenizer class
 */
export class Tokenizer {
  #id;
  constructor(id: number) {
    this.#id = id;
  }

  /**
   * Get the vocab size
   */
  getVocabSize(withAddedTokens = true) {
    return wasm_tokenizer_get_vocab_size(this.#id, withAddedTokens);
  }

  /**
   * Get the vocab
   */
  getVocab(withAddedTokens = true) {
    return wasm_tokenizer_get_vocab(this.#id, withAddedTokens);
  }

  /**
   * Get the token from an id
   */
  idToToken(id: number) {
    return wasm_tokenizer_id_to_token(this.#id, id);
  }

  /**
   * Get the id from a token
   */
  tokenToId(token: string) {
    return wasm_tokenizer_token_to_id(this.#id, token);
  }
  
  /**
   * Encode a sentence
   * @param sentence sentence to tokenize
   * @returns
   */
  encode(sentence: string) {
    return wasm_tokenizer_encode(this.#id, sentence);
  }

  /**
   * Decode a sentence
   * @param tokens tokens to decode
   * @returns
   */
  decode(ids: Uint32Array, skipSpecialTokens = false) {
    return wasm_tokenizer_decode(this.#id, ids, skipSpecialTokens);
  }

  /**
   * Save the tokenizer as json
   */
  save(): string;
  /**
   * Save the tokenizer as json
   * @param pretty pretty print the json
   */
  save(pretty: boolean): string;
  save(...args: [boolean?]) {
    return wasm_tokenizer_save(this.#id, args[0] ?? false);
  }
  /**
   * Load a tokenizer from json data
   * @param json string
   * @returns
   */
  static fromJSON(json: string) {
    return new Tokenizer(wasm_tokenizer_from_json(json));
  }
}
