/** Map discrete values into numbers */
export class DiscreteMapper<T> {
  /** Map categories to indices */
  mapping: Map<T, number>;
  /** An internal counter for remembering the last index in mapping. */
  #lastToken: Uint32Array;
  constructor() {
    this.mapping = new Map();
    this.#lastToken = new Uint32Array(1);
  }
  /** Construct a mapping from a given set of text. */
  fit(targets: T[]): this {
    let i = 0;
    while (i < targets.length) {
      if (!this.mapping.has(targets[i])) {
        const token = this.#incrementToken();
        this.mapping.set(targets[i], token);
      }
      i += 1;
    }
    return this;
  }
  /**
   * Encode values into their respective mappings.
   * Returns -1 in case of missing mapping.
   */
  transform(targets: T[]): number[] {
    const res = new Array(targets.length);
    let i = 0;
    while (i < targets.length) {
      const index = this.mapping.get(targets[i]) ?? -1;
      res[i] = index;
      i += 1;
    }
    return res;
  }
  /** Convert mapped numbers into actual values */
  untransform(data: number[]): T[] {
    const res = new Array(data.length);
    for (let i = 0; i < res.length; i += 1) {
      res[i] = this.getOg(data[i]) || "__unknown__";
    }
    return res;
  }
  getOg(data: number): T | undefined {
    for (const [k, v] of this.mapping.entries()) {
      if (v === data) {
        return k;
      }
    }
    return undefined;
  }
  #incrementToken(): number {
    return Atomics.add(this.#lastToken, 0, 1);
  }
}
