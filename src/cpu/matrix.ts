import { DataArray, DataType } from "../../deps.ts";
import { fromType, getType } from "../util.ts";

export class CPUMatrix<T extends DataType = DataType> {
    constructor(
        public data: DataArray<T>,
        public x: number,
        public y: number,
        public type: DataType = getType(data),
    ) { }

    static with(
        x: number,
        y: number,
        type: DataType,
    ) {
        const data = new (fromType(type))(x * y);
        return new this(data, x, y)
    }

    static add(matA: CPUMatrix, matB: CPUMatrix) {
        const res = CPUMatrix.with(matA.x, matA.y, matA.type)
        for (let i = 0; i < matA.data.length; i++) {
            res.data[i] = matA.data[i] + matB.data[i]
        }
        return res
    }

    static mul(matA: CPUMatrix, matB: CPUMatrix) {
        const res = CPUMatrix.with(matB.x, matA.y, matA.type)
        for (let x = 0; x < matB.x; x++) {
            for (let y = 0; y < matA.y; y++) {
                let sum = 0
                for (let k = 0; k < matA.x; k++) {
                    const a = k + y * matA.x;
                    const b = x + k * matB.x;
                    sum += matA.data[a] * matB.data[b];
                }
                const idx = x + y * matB.x;
                res.data[idx] = sum;
            }
        }
        return res
    }
    static sub(matA: CPUMatrix, matB: CPUMatrix) {
        const res = CPUMatrix.with(matA.x, matA.y, matA.type)
        for (let i = 0; i < matA.data.length; i++) {
            res.data[i] = matA.data[i] - matB.data[i]
        }
        return res
    }
    static reduce(mat: CPUMatrix, func: (acc: number, val: number) => number) {
        for (let i = 0; i < mat.data.length; i++) {
            mat.data[0] = func(mat.data[0], mat.data[i])
        }
        return mat;

    }
    static mirror(mat: CPUMatrix) {
        for (let i = 0; i < mat.data.length; i++) {
            mat.data[i] = mat.data[mat.data.length - i - 1]
        }
        return mat;
    }
    static transpose(mat: CPUMatrix) {
        const resMat = CPUMatrix.with(mat.y, mat.x, mat.type)
        for (let x = 0; x < mat.x; x++) {
            for (let y = 0; y < mat.y; y++) {
                const idx = x + y * mat.x;
                resMat.data[idx] = mat.data[y + x * mat.y]
            }
        }
        return resMat;
    }

    fill(val: number) {
        this.data.fill(val)
    }
}