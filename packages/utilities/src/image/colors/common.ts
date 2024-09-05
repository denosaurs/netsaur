import type { Image } from "../../utils/mod.ts";
import type { Pixel } from "../../utils/common_types.ts";
import type { ColorHistogram } from "./histogram.ts";

export function getAverageColor(
  vbox: ColorRange,
  histo: ColorHistogram,
): Pixel {
  let total = 0;
  let totalR = 0, totalG = 0, totalB = 0;
  let ri = vbox.r.min;
  while (ri <= vbox.r.max) {
    let gi = vbox.g.min;
    while (gi <= vbox.g.max) {
      let bi = vbox.b.min;
      while (bi <= vbox.b.max) {
        const count = histo.getQuantized([ri, gi, bi]) || 0;
        total += count;
        totalR += count * (ri + 0.5) * 8;
        totalG += count * (gi + 0.5) * 8;
        totalB += count * (bi + 0.5) * 8;
        bi += 1;
      }
      gi += 1;
    }
    ri += 1;
  }
  if (total) {
    return [
      ~~(totalR / total),
      ~~(totalG / total),
      ~~(totalB / total),
      255,
    ];
  }
  // In case box is empty
  return [
    Math.trunc(8 * (vbox.r.min + vbox.r.max + 1) / 2),
    Math.trunc(8 * (vbox.g.min + vbox.g.max + 1) / 2),
    Math.trunc(8 * (vbox.b.min + vbox.b.max + 1) / 2),
    255,
  ];
}

/** The vbox */
export interface ColorRange {
  r: { min: number; max: number };
  g: { min: number; max: number };
  b: { min: number; max: number };
}

/** Get the minimum and maximum RGB values. */
export function getColorRange(
  image: Image,
  sigBits = 5,
): ColorRange {
  const quantizeBy = 8 - sigBits;
  const range = {
    r: { min: 1000, max: 0 },
    g: { min: 1000, max: 0 },
    b: { min: 1000, max: 0 },
  };
  let i = 0;
  while (i < image.pixels) {
    const pixel = image.getNthPixel(i).map((x) => x ? x >> quantizeBy : 0);
    if (pixel[0] < range.r.min) {
      range.r.min = pixel[0];
    }
    if (pixel[0] > range.r.max) {
      range.r.max = pixel[0];
    }

    if (pixel[1] < range.g.min) {
      range.g.min = pixel[1];
    }
    if (pixel[1] > range.g.max) {
      range.g.max = pixel[1];
    }

    if (pixel[2] < range.b.min) {
      range.b.min = pixel[2];
    }
    if (pixel[2] > range.b.max) {
      range.b.max = pixel[2];
    }

    i += 1;
  }
  return range;
}
