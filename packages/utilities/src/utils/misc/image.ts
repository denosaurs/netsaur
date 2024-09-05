import type { Pixel } from "../common_types.ts";

export type ImageData = {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  channels: number; // always 4, for compat with retraigo/vectorizer
  colorSpace: "srgb" | "display-p3";
};

type ImageOptions = {
  data: Uint8ClampedArray;
  width: number;
  height?: number;
  channels?: number;
};

export class Image implements ImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  channels: number; // always 4, for compat with retraigo/vectorizer
  colorSpace: "srgb" | "display-p3";
  constructor(data: ImageOptions) {
    this.data = Uint8ClampedArray.from(data.data);
    // N-channels is always 4
    this.channels = 4;
    this.width = data.width;
    this.height = data.height ??
      this.data.length / (this.width * this.channels);
    // If height is not an integer or width is incorrect
    if (this.height !== ~~this.height) {
      throw new TypeError(
        `Height must be an integer. Received ${this.height}.`,
      );
    }
    // Only srgb is supported
    this.colorSpace = "srgb";
  }
  get pixels(): number {
    return this.width * this.height;
  }
  getNthPixel(n: number): [number, number, number, number] {
    const offset = n << 2;
    return [
      this.data[offset],
      this.data[offset + 1],
      this.data[offset + 2],
      this.data[offset + 3],
    ];
  }
  getPixel(row: number, col: number): Pixel {
    if (row >= this.height) {
      throw new RangeError(
        `Requested row ${row} is outside of bounds 0..${this.height}.`,
      );
    }
    if (col >= this.width) {
      throw new RangeError(
        `Requested column ${col} is outside of bounds 0..${this.width}.`,
      );
    }
    const offset = row * this.width + col;
    const [r, g, b, a] = this.data.slice(offset, offset + 4);
    return [r, g, b, a];
  }
  setPixel(row: number, col: number, [r, g, b, a]: Pixel) {
    if (row >= this.height) {
      throw new RangeError(
        `Requested row ${row} is outside of bounds 0..${this.height}.`,
      );
    }
    if (col >= this.width) {
      throw new RangeError(
        `Requested column ${col} is outside of bounds 0..${this.width}.`,
      );
    }
    const offset = row * this.width + col;
    this.data.set(typeof a !== "undefined" ? [r, g, b, a] : [r, g, b], offset);
  }
  updatePixel(
    row: number,
    col: number,
    color: Pixel,
  ) {
    if (row >= this.height) {
      throw new RangeError(
        `Requested row ${row} is outside of bounds 0..${this.height}.`,
      );
    }
    if (col >= this.width) {
      throw new RangeError(
        `Requested column ${col} is outside of bounds 0..${this.width}.`,
      );
    }
    const offset = row * this.width + col;
    for (let i = 0; i < color.length; i += 1) {
      this.data[offset + i] += color[i] ?? 0;
    }
  }
}
