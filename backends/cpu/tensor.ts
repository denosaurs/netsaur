import { TensorBackend, TensorLike, Tensor2DCPU, TypedArray } from "../../core/types.ts";
import { flatten } from "../../core/util.ts";
import { CPUMatrix } from "./matrix.ts"


export class TensorCPUBackend implements TensorBackend{
    tensor2D(values: TensorLike, width: number, height: number): Tensor2DCPU {
        return new CPUMatrix(new Float32Array(flatten(values as TypedArray) as ArrayBufferLike), width, height)
    }
    tensor1D(values: TensorLike): Float32Array {
        return new Float32Array(values as TypedArray);
    }
}