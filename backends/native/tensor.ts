import { Tensor2DNative, TensorBackend, TensorLike } from "../../core/types.ts";
import { Matrix } from "./matrix.ts";

export class TensorNativeBackend implements TensorBackend {
  constructor() {
  }
  tensor2D(
    values: TensorLike,
    width: number,
    height: number,
  ): Tensor2DNative {
    return new Matrix<"f32">(
      height,
      width,
      new Float32Array((values as number[][]).flat()),
    );
  }
  tensor1D(values: TensorLike) {
    return new Matrix(
      (values as number[]).length,
      1,
      new Float32Array(values as number[]),
    );
  }
}
