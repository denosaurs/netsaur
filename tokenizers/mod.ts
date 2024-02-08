import {
  instantiate,
  wasm_tokenizer_decode,
  wasm_tokenizer_encode,
  wasm_tokenizer_from_json,
  wasm_tokenizer_get_vocab,
  wasm_tokenizer_get_vocab_size,
  wasm_tokenizer_id_to_token,
  wasm_tokenizer_save,
  wasm_tokenizer_token_to_id,
} from "./lib/netsaur_tokenizers.generated.js";

let initialized = false;
export async function init(): Promise<void> {
  if (initialized) return;
  await instantiate({
    url: new URL(import.meta.url).protocol !== "file:"
      ? new URL(
        "https://github.com/denosaurs/netsaur/releases/download/0.3.1/netsaur_tokenizers_bg.wasm",
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
  #id: number;
  constructor(id: number) {
    this.#id = id;
  }

  /**
   * Get the vocab size
   */
  getVocabSize(withAddedTokens = true): number {
    return wasm_tokenizer_get_vocab_size(this.#id, withAddedTokens);
  }

  /**
   * Get the vocab
   */
  // deno-lint-ignore no-explicit-any
  getVocab(withAddedTokens = true): any {
    return wasm_tokenizer_get_vocab(this.#id, withAddedTokens);
  }

  /**
   * Get the token from an id
   */
  idToToken(id: number): string {
    return wasm_tokenizer_id_to_token(this.#id, id);
  }

  /**
   * Get the id from a token
   */
  tokenToId(token: string): number {
    return wasm_tokenizer_token_to_id(this.#id, token);
  }

  /**
   * Encode a sentence to tokens
   * @param sentence sentence to tokenize
   * @returns
   */
  encode(sentence: string): Uint32Array {
    return wasm_tokenizer_encode(this.#id, sentence);
  }

  /**
   * Decode a sentence from its encoded tokens to a string
   * @param tokens tokens to decode
   * @returns
   */
  decode(ids: Uint32Array, skipSpecialTokens = false): string {
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
  save(...args: [boolean?]): string {
    return wasm_tokenizer_save(this.#id, args[0] ?? false);
  }
  /**
   * Load a tokenizer from json data
   * @param json string
   * @returns
   */
  static fromJSON(json: string): Tokenizer {
    return new Tokenizer(wasm_tokenizer_from_json(json));
  }
}
