import { DataArrayConstructor, DataType, WebGPUData } from "../../deps.ts";

export class GPUMatrix<T extends DataType = DataType> {
    public type: DataType
    constructor(
        public data: WebGPUData<T>,
        public x: number,
        public y: number,
        type?: DataType,
    ) {
        this.type = type ?? (
            data instanceof Uint32Array ? "u32"
          : data instanceof Int32Array ? "i32"
          : data instanceof Float32Array ? "f32"
          : undefined
        ) as T
    }

    static with(
        x: number,
        y: number,
        type: DataType,
    ) {
        const data = new DataArrayConstructor[type](x * y);
        return new this(data, x, y)
    }
  
    // static mul(matA: GPUMatrix, matB: GPUMatrix) {
    //     // imean im just saying, we shouldnt have a GPUMatrix.mul, but rather a specific function such as 
    //     // strings is easier, since the snabel already handled the gpu compiling thing in neo
    //     matMul(backend)
    //     return res
    // }
    
    // static reduce(mat: GPUMatrix, func: (acc: number, val: number) => number) {
    //     for (let i = 0; i < mat.data.length; i++) {
    //         mat.data[0] = func(mat.data[0], mat.data[i])
    //     }
    //     return mat;

    // }
}