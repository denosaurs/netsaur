/** Mean Absolute Error */
export function mae(y: ArrayLike<number>, y1: ArrayLike<number>): number {
  let err = 0;
  for (let i = 0; i < y.length; i += 1) {
    err += (y[i] - y1[i]) ** 2;
  }
  return err / y.length;
}

/** Mean Square Error */
export function mse(y: ArrayLike<number>, y1: ArrayLike<number>): number {
  let err = 0;
  for (let i = 0; i < y.length; i += 1) {
    err += (y[i] - y1[i]) ** 2;
  }
  return err / y.length;
}

/** Root Mean Square Error */
export function rmse(y: ArrayLike<number>, y1: ArrayLike<number>): number {
  return Math.sqrt(mse(y, y1));
}

/** R2 Score for regression */
export function r2(y: ArrayLike<number>, y1: ArrayLike<number>): number {
  let mean = 0;
  for (let i = 0; i < y.length; i += 1) {
    mean += y[i];
  }
  mean /= y.length;
  let ssr = 0, sst = 0;
  for (let i = 0; i < y.length; i += 1) {
    ssr += Math.pow(y1[i] - mean, 2);
    sst += Math.pow(y[i] - mean, 2);
  }
  return ssr / sst;
}
