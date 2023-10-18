import {
  instantiate,
  wasm_tokenizer_from_json,
  wasm_tokenizer_save,
  wasm_tokenizer_tokenize,
} from "./lib/netsaur_tokenizers.generated.js";

let initialized = false;
export async function init() {
  if (initialized) return;
  await instantiate({
    url: new URL(import.meta.url).protocol !== "file:"
      ? new URL(
        "https://github.com/denosaurs/netsaur/releases/download/0.2.12/netsaur_tokenizers_bg.wasm",
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
   * Tokenize a sentence
   * @param sentence sentence to tokenize
   * @returns
   */
  tokenize(sentence: string) {
    return wasm_tokenizer_tokenize(this.#id, sentence);
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
