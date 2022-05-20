// deno-lint-ignore no-unused-vars
import { WebGPUData } from "../../deps.ts";

export interface GPUCostFunction {
  /** Return the cost associated with an output `a` and desired output `y`. */
  cost(type: string): string;

  /** Return the error delta from the output layer. */
  prime(type: string): string;
}

/**
 * Cross entropy cost function is the standard cost function for binary classification.
 */
export class CrossEntropy implements GPUCostFunction {
  cost(type: string) {
    return `var sum: ${type} = ${type}(0);
    for (var i = ${type}(0); i < yHat.length; i++) {
      sum += -y[i] * log(yHat[i]) - (${type}(1) - y[i]) * log(${type}(1) - yHat[i]);
    }
    return sum;`;
  }

  prime(_type: string) {
    return `return yHat - y`;
  }
}

/**
 * Hinge cost function is the standard cost function for multiclass classification.
 */
export class Hinge implements GPUCostFunction {
  cost(type: string) {
    return `var max: ${type} = ${type}(0);
    for (var i = ${type}(0); i < yHat.length; i++) {
      var value: ${type} = y[i] - (1 - 2 * y[i]) * yHat[i];
      if (value > max) { max = value; }
    }
    return max;`;
  }

  prime(type: string) {
    return `if (y * yHat * ${type}(2) < ${type}(1)) { return -y * yHat; } return 0;`;
  }
}
