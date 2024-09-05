/**
 * Get n evenly distributed numbers in a range.
 * @param n Number of numbers to generate.
 * @param min Lower limit of range (inclusive).
 * @param max Upper limit of range (exclusive).
 * @returns Array of n evenly distributed numbers.
 */
export function useRange(n: number, min = 0, max = 1): number[] {
  const res = new Array(n);
  let i = 0;
  while (i < n) {
    res[i] = min + ((i * (max - min)) / n);
    i += 1;
  }
  return res;
}

/**
 * Get an array of numbers between a given range,
 * incremented by a step.
 * @param min Lower limit of range (inclusive).
 * @param max Upper limit of range (exclusive).
 * @param step step to increment by
 * @returns Array of numbers
 */
export function useSeries(max: number): number[];
export function useSeries(min: number, max: number, step?: number): number[];
export function useSeries(min: number, max?: number, step = 1): number[] {
  if (typeof max === "undefined") [min, max] = [0, min];
  const res = new Array(~~((max - min) / step));
  res[0] = min;
  let i = 1;
  while (i < res.length) {
    res[i] = res[i - 1] + step;
    i += 1;
  }
  return res;
}
