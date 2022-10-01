import { DataType, DataTypeArray } from "../../deps.ts";
import { iterate1D } from "../util.ts";

export interface CPUCostFunction<T extends DataType = DataType> {
  /** Return the cost associated with an output `a` and desired output `y`. */
  cost(yHat: DataTypeArray<T>, y: DataTypeArray<T>): number;

  /** Return the error delta from the output layer. */
  prime(yHat: number, y: number): number;
}

/**
 * Cross entropy cost function is the standard cost function for binary classification.
 */
export class CrossEntropy<T extends DataType = DataType>
  implements CPUCostFunction {
  cost(yHat: DataTypeArray<T>, y: DataTypeArray<T>) {
    let sum = 0;
    iterate1D(yHat.length, (i: number) => {
      sum += -y * Math.log(yHat[i]) - (1 - y[i]) * Math.log(1 - yHat[i]);
    });
    return sum;
  }

  prime(yHat: number, y: number) {
    return yHat - y;
  }
}

/**
 * Hinge cost function is the standard cost function for multiclass classification.
 */
export class Hinge<T extends DataType = DataType> implements CPUCostFunction {
  cost(yHat: DataTypeArray<T>, y: DataTypeArray<T>) {
    let max = -Infinity;
    iterate1D(yHat.length, (i: number) => {
      const value = y[i] - (1 - 2 * y[i]) * yHat[i];
      if (value > max) max = value;
    });
    return max;
  }

  prime(yHat: number, y: number) {
    return y * yHat * 2 < 1 ? -y * yHat : 0;
  }
}
