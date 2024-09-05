/**
 * Image-related utilities for machine learning.
 * @module
 */

import { Image } from "../../utils/mod.ts";
import type { Pixel } from "../../utils/common_types.ts";
import { quantizeByMedianCut } from "./median_cut.ts";

/** Extract colors from an image. */
export function extractColors(image: Image, nColors: number): Pixel[] {
  return quantizeByMedianCut(image, nColors, 5);
}

export { getHistogram } from "./histogram.ts";
export { Image };
