import type { Sliceable } from "../common_types.ts";
import { Matrix } from "../mod.ts";
import { useShuffle } from "../random/shuffle.ts";

interface SplitOptions {
  ratio: [number, number];
  shuffle?: boolean;
}

/** Split arrays by their first axis. Only usable on Matrices and Arrays. */
export function useSplit<T extends Sliceable[]>(
  options: SplitOptions = { ratio: [7, 3], shuffle: false },
  ...arr: T
): [typeof arr, typeof arr] {
  if (!arr.every((x) => x.length === arr[0].length)) {
    throw new Error("All arrays must have equal length!");
  }
  const { ratio, shuffle } = options;
  const idx = Math.floor(arr[0].length * (ratio[0] / (ratio[0] + ratio[1])));
  if (!shuffle) {
    return [arr.map((x) => x.slice(0, idx)), arr.map((x) => x.slice(idx))] as [
      T,
      T
    ];
  } else {
    const shuffled = useShuffle(0, arr[0].length);
    const x1 = shuffled.slice(0, idx);
    const x2 = shuffled.slice(idx);
    return [
      arr.map((x) => {
        if (x instanceof Matrix) {
          return x.rowSelect(x1);
        } else {
          return x1.map((idx) => (x as unknown[])[idx]);
        }
      }) as unknown as typeof arr,
      arr.map((x) => {
        if (x instanceof Matrix) {
          return x.rowSelect(x2);
        } else {
          return x2.map((idx) => (x as unknown[])[idx]);
        }
      }) as unknown as typeof arr,
    ];
  }
}
