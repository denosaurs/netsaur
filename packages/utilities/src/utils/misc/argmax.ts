export function argmax(mat: ArrayLike<number | bigint>): number {
  let max = mat[0];
  let index = 0;
  for (let i = 0; i < mat.length; i++) {
    if (mat[i] > max) {
      max = mat[i] as typeof max;
      index = i;
    }
  }
  return index;
}
