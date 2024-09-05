const TWO_PI = Math.PI * 2;

/**
 * A **Normal** or **Gaussian** distribution is a type of
 * continuous probability distribution dependent on two
 * parameters:
 *
 * **μ** - The **mean**
 * **σ** - The **standard deviation**
 *
 * This implementation makes use of the popular Box-Muller transform.
 */

/**
 * Generate a normal random variate.
 * @param mean Mean of the distribution μ.
 * @param stddev Standard Deviation of the distribution σ.
 * @returns A normal random variate.
 */
export function useNormal(
  mean: number,
  stddev: number,
): [number, number] {
  const u = [Math.random(), Math.random()];

  const m = Math.sqrt(-2.0 * Math.log(u[0]));
  return [
    (stddev * m * Math.cos(TWO_PI * u[1])) + mean,
    (stddev * m * Math.sin(TWO_PI * u[1])) + mean,
  ];
}

/**
 * Generate a normally distributed array.
 * @param mean Mean of the distribution μ.
 * @param variance Variance of the distribution σ^2.
 * @returns A normally distributed array.
 */

export function useNormalArray(
  num: number,
  mean: number,
  variance: number,
): Float32Array {
  const result = new Float32Array(num);
  let i = 0;
  const stddev = Math.sqrt(variance);
  while (i < num) {
    result[i] = useNormal(mean, stddev)[0];
    ++i;
  }
  return result;
}
