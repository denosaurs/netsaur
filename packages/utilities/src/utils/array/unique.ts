/**
 * Remove duplicate values in an array.
 * Uses a strict = for identifying duplicates.
 * @param {T[]} arr Array to remove duplicates from.
 * @returns {T[]} Array with only unique elements.
 */
export function useUnique<T>(arr: ArrayLike<T>): T[] {
  const array = Array.from(arr);
  return array.filter((x, i) => array.indexOf(x) === i);
}
