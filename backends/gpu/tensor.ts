import {
  Tensor2DGPU,
  TensorBackend,
  TensorLike,
  TypedArray,
} from "../../core/types.ts";
import { flatten } from "../../core/util.ts";
import { WebGPUBackend } from "../../deps.ts";
import { GPUMatrix } from "./matrix.ts";

export class TensorGPUBackend implements TensorBackend {
  constructor(public backend: WebGPUBackend) {
  }
  async tensor2D(
    values: TensorLike,
    width: number,
    height: number,
  ): Promise<Tensor2DGPU> {
    return await GPUMatrix.from(
      this.backend,
      // deno-lint-ignore no-explicit-any
      new Float32Array(flatten(values as TypedArray) as any),
      width,
      height,
    );
  }
  tensor1D(values: TensorLike): Float32Array {
    return new Float32Array(values as TypedArray);
  }
}
