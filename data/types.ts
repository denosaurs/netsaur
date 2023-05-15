// deno-lint-ignore-file no-explicit-any
import { Tensor } from "../mod.ts";

export interface DataLike {
  /**
   * Model input data
   */
  train_x: Tensor<any>;

  /**
   * Model output data / labels
   */
  train_y: Tensor<any>;

  /**
   *  Model test input data
   */
  test_x?: Tensor<any>;

  /**
   * Model test output data / labels
   */
  test_y?: Tensor<any>;
}
