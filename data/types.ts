import type { Rank, Tensor } from "../mod.ts";

export interface DataLike {
  /**
   * Model input data
   */
  train_x: Tensor<Rank>;

  /**
   * Model output data / labels
   */
  train_y: Tensor<Rank>;

  /**
   *  Model test input data
   */
  test_x?: Tensor<Rank>;

  /**
   * Model test output data / labels
   */
  test_y?: Tensor<Rank>;
}
