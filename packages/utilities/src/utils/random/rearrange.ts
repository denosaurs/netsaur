import { useShuffle } from "./shuffle.ts";

/**
 * Rearrange characters in a string randomly.
 * @param {number|string} n Number / String to rearrange.
 * @returns {number|string} Number / String rearranged randomly.
 */
export function useRearrange(n: number | string): number | string {
  const res = (typeof n === "number" ? n.toString() : n).split("");
  const shuffled = useShuffle(res).join("");
  return typeof n === "number" ? Number(shuffled) : shuffled;
}
export default useRearrange;
