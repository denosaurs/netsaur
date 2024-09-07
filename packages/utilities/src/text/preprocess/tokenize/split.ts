import { DefaultIgnoreList } from "../../../constants/stop_words.ts";
import type { BaseTokenizerOptions } from "../../../utils/common_types.ts";

/** Tokenize text based on separator (whitespace) */
export class SplitTokenizer {
  /** Words to ignore from vocabulary */
  skipWords: "english" | false | string[];
  /** Configuration / Function for preprocessing */
  vocabulary: Map<string, number>;
  /** An internal counter for remembering the last index in vocabulary. */
  #lastToken: Uint32Array;
  constructor(
    options: Partial<BaseTokenizerOptions & { indices: boolean }> = {},
  ) {
    this.skipWords = options.skipWords ?? false;
    this.vocabulary = options.vocabulary ?? new Map();
    this.#lastToken = new Uint32Array(1);
    if (options.indices && !this.vocabulary.size) {
      this.#lastToken[0] = 2;
      this.vocabulary.set("__pad__", 0);
      this.vocabulary.set("__unk__", 1);
    }
    if (this.vocabulary.size) {
      this.#lastToken[0] = this.vocabulary.size;
    }
  }
  get lastToken(): number {
    return Atomics.load(this.#lastToken, 0);
  }
  /** Construct a vocabulary from a given set of text. */
  fit(text: string | string[]): this {
    if (Array.isArray(text)) {
      let i = 0;
      while (i < text.length) {
        this.fit(text[i]);
        i += 1;
      }
    } else {
      const words = this.split(text);
      let i = 0;
      while (i < words.length) {
        if (!this.vocabulary.has(words[i])) {
          if (this.skipWords === "english") {
            if (DefaultIgnoreList.includes(words[i])) {
              i += 1;
              continue;
            }
          } else if (Array.isArray(this.skipWords)) {
            if (this.skipWords.includes(words[i])) {
              i += 1;
              continue;
            }
          }
          const token = this.#incrementToken();
          this.vocabulary.set(words[i], token);
        }
        i += 1;
      }
    }
    return this;
  }
  #incrementToken(): number {
    return Atomics.add(this.#lastToken, 0, 1);
  }
  /**
   * Convert a document (string | array of strings) into vectors.
   */
  transform(text: string | string[]): number[][] {
    if (!this.vocabulary.size) {
      throw new Error(
        "Tokenizer vocabulary not initialized yet. Call `Tokenizer()` with a custom vocabulary or use `.fit()` on text.",
      );
    }
    if (Array.isArray(text)) {
      const size = Math.max(...text.map((x) => this.split(x).length));
      const res = Array(text.length);
      let i = 0;
      while (i < text.length) {
        res[i] = this.#transform(text[i], size);
        i++;
      }
      return res;
    }
    return [this.#transform(text, 0)];
  }
  #transform(text: string, size: number): number[] {
    const words = this.split(text);
    if (!size) size = words.length;
    const res = new Array(size);
    res.fill(this.vocabulary.get("__pad__") || 0);
    let i = 0;
    while (i < words.length && i < size) {
      if (this.vocabulary.has(words[i])) {
        const index = this.vocabulary.get(words[i]);
        res[i] = typeof index === "number"
          ? index
          : this.vocabulary.get("__unk__") || 0;
      } else {
        res[i] = this.vocabulary.get("__unk__") || 0;
      }
      i++;
    }
    return res;
  }
  // TODO: Support custom split modes
  split(text: string): string[] {
    return text.split(" ");
  }
}
