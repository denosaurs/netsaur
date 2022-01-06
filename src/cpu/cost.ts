import { DataArray, DataType } from "../../deps.ts";

export interface CPUCostFunction<T extends DataType = DataType> {
    /** Return the cost associated with an output `a` and desired output `y`. */
    cost(yHat: DataArray<T>, y: DataArray<T>): number

    /** Return the error delta from the output layer. */
    measure(z: number, yHat: number, y: number): number
}

export class CrossEntropy<T extends DataType = DataType> implements CPUCostFunction {
    public cost(yHat: DataArray<T>, y: DataArray<T>) {
        let sum = 0
        for (let i = 0; i < yHat.length; i++) {
            sum += -y * Math.log(yHat[i]) - (1 - y[i]) * Math.log(1 - yHat[i])
        }
        return sum
    }

    public measure(_: number, yHat: number, y: number) {
        return yHat - y
    }
}

export class Hinge<T extends DataType = DataType> implements CPUCostFunction {
    public cost(yHat: DataArray<T>, y: DataArray<T>) {
        let max = -Infinity
        for (let i = 0; i < yHat.length; i++) {
            const value = y[i] - (1 - 2 * y[i]) * yHat[i]
            if (value > max) max = value
        }
        return max
    }

    public measure(z: number, yHat: number, y: number) {
        return y * yHat * z < 1 ? -y * yHat : 0
    }
}
