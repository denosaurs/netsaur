import type { Image } from "../../utils/mod.ts";
import type { Patch2d, PatchCollection } from "../../utils/common_types.ts";

/**
 * Extract patches from a 2d image.
 */

/**
 * Get the row and column of the nth element of a
 * 2d array.
 * @param n Current Position
 * @param width Width of a row
 * @returns [row, column]
 */
function clamp(n: number, width: number): [number, number] {
  if (n >= width) return [~~(n / width), n % width];
  return [0, n];
}

/** Private function to extract patches */
function extract(image: Image, options: Patch2d): [Uint8ClampedArray, number] {
  /** Get number of possible patches in each dimension */
  const nX = image.width - options.width + 1;
  const nY = image.height - options.height + 1;
  /** Total number of patches is nX * nY */
  const nPatches = nX * nY;
  /** Area of each patch, used to calculate offset of resulting array */
  const patchArea = options.width * options.height;

  const res = new Uint8ClampedArray(
    options.width * options.height * image.channels * nPatches,
  );

  let i = 0;
  while (i < nPatches) {
    const [row, col] = clamp(i, nX);
    /** Starting index of the current patch */
    const offset = row * image.width + col;
    let j = 0;
    while (j < options.height) {
      /** Starting index of the current subrow */
      const patchRow = j * image.width;
      /** Copy the entire patchArea-size patch into the resulting array */
      res.set(
        image.data.slice(
          (offset + patchRow) * image.channels,
          (offset + options.width + patchRow) * image.channels,
        ),
        (i * patchArea + j * options.width) * image.channels,
      );
      j += 1;
    }
    i += 1;
  }
  return [res, nPatches];
}

/**
 * Extract patches from a 2d image
 * @param image Source image to extract patches from
 * @param options Dimensions of a single patch
 * @returns A collection of patches as a single Uint8ClampedArray
 */
export function patch2d(image: Image, options: Patch2d): PatchCollection {
  if (image.width < options.width) {
    throw new Error("Patch width cannot be greater than image width.");
  }
  if (image.height < options.height) {
    throw new Error("Patch height cannot be greater than image width.");
  }

  const [patches, n] = extract(image, options);

  return {
    width: options.width,
    height: options.height,
    channels: image.channels,
    size: n,
    data: patches,
  };
}
