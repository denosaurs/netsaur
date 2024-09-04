import type { Image } from "../../utils/mod.ts";
import type { Pixel } from "../../utils/common_types.ts";

/**
 * Histogram of colors with reduced space
 * Effectively quantizes the image into 32768 colors
 */
export class ColorHistogram {
  #data: Uint32Array;
  #quantizeBy: number;
  sigBits: number;
  constructor(sigBits: number) {
    this.sigBits = sigBits;
    this.#quantizeBy = 8 - sigBits;
    this.#data = new Uint32Array(1 << (sigBits * 3));
  }
  #getIndex([r, g, b]: [number, number, number, number?]) {
    // ignore alpha
    const index = ((r >> this.#quantizeBy) << (this.sigBits << 1)) +
      ((g >> this.#quantizeBy) << this.sigBits) +
      (b >> this.#quantizeBy);
    return index;
  }
  get(color: Pixel): number {
    const index = this.#getIndex(color);
    return this.#data[index];
  }
  getQuantized(color: Pixel): number {
    const index = (color[0] << 10) + (color[1] << 5) + color[2];
    return this.#data[index];
  }
  add(color: Pixel, amount: number): number {
    const index = this.#getIndex(color);
    return Atomics.add(this.#data, index, amount);
  }
  get raw(): Uint32Array {
    return this.#data;
  }
  get length(): number {
    return this.#data.filter((x) => x).length;
  }
  static getColor(index: number, sigBits: number): Pixel {
    const quantizeBy = 8 - sigBits;
    const ri = index >> 10;
    const gi = (index - (ri << 10)) >> 5;
    const bi = index - (ri << 10) - (gi << 5);
    return [ri << quantizeBy, gi << quantizeBy, bi << quantizeBy, 255];
  }
}

/** Get a histogram of frequency of colors. */
export function getHistogram(image: Image, sigBits = 5): ColorHistogram {
  const histo = new ColorHistogram(sigBits);
  let i = 0;
  while (i < image.pixels) {
    const hIndex = image.getNthPixel(i);
    histo.add(hIndex, 1);
    i += 1;
  }
  return histo;
}
