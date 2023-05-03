import { Rank, Shape, Tensor, TensorData } from "../../mod.ts";
import { length } from "../core/tensor/util.ts";
import { BackendType } from "../core/types.ts";

/**
 * CPU Tensor Backend.
 */
export class CPUTensorBackend {
  zeroes<R extends Rank, B extends BackendType>(shape: Shape[R]): Tensor<R, B> {
    return new Tensor(new Float32Array(length(shape)) as TensorData[B], shape);
  }

  from<R extends Rank, B extends BackendType>(
    values: Float32Array,
    shape: Shape[R],
  ): Tensor<R, B> {
    return new Tensor(values as TensorData[B], shape);
  }

  //deno-lint-ignore require-await
  async get(tensor: Tensor<Rank, BackendType>) {
    return tensor.data as Float32Array;
  }

  set(tensor: Tensor<Rank, BackendType>, values: Float32Array) {
    tensor.data = values;
  }
}
