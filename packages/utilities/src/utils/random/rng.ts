/**
 * Get random number from range
 * @param min Min value of range.
 * @param max Max value of range.
 * @param allowDecimals Whether to allow decimal point in result.
 * @returns Random number
 */
export function useRNG(
  min: number,
  max: number,
  allowDecimals = false,
): number {
  const rng = min + (Math.random() * (max - min));
  return allowDecimals ? rng : Math.round(rng);
}
