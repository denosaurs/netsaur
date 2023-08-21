import {
  instantiate,
  wasm_tokenizer_from_json,
  wasm_tokenizer_tokenize,
} from "./lib/netsaur_tokenizers.generated.js";

let initialized = false;
export async function init() {
  if (initialized) return;
  await instantiate({
    url: new URL(import.meta.url).protocol !== "file:"
      ? new URL(
        "https://github.com/denosaurs/netsaur/releases/download/0.2.10/netsaur_tokenizers_bg.wasm",
        import.meta.url,
      )
      : undefined,
  });
  initialized = true;
}

export class Tokenizer {
  #id;
  constructor(id: number) {
    this.#id = id;
  }

  tokenize(sentence: string) {
    return wasm_tokenizer_tokenize(this.#id, sentence);
  }

  static fromJson(json: string) {
    return new Tokenizer(wasm_tokenizer_from_json(json));
  }
}
