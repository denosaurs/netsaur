// deno-lint-ignore no-unused-vars
import { WebGPUData } from "../../deps.ts";

export interface GPUCostFunction {
  name: string;
  /** Return the cost associated with an output `a` and desired output `y`. */
  cost(...values: string[]): string;

  /** Return the error delta from the output layer. */
  prime(...values: string[]): string;
}

/**
 * Cross entropy cost function is the standard cost function for binary classification.
 */
export class CrossEntropy implements GPUCostFunction {
  name = "crossentropy"
  cost() {
    return `var sum: f32 = f32(0);
    for (var i = f32(0); i < yHat.length; i++) {
      sum += -y[i] * log(yHat[i]) - (f32(1) - y[i]) * log(f32(1) - yHat[i]);
    }
    return sum;`;
  }

  prime() {
    return `return yHat - y`;
  }
}

/**
 * Hinge cost function is the standard cost function for multiclass classification.
 */
export class Hinge implements GPUCostFunction {
  name = "hinge"
  cost() {
    return `var max: f32 = f32(0);
    for (var i = f32(0); i < yHat.length; i++) {
      var value: f32 = y[i] - (1 - 2 * y[i]) * yHat[i];
      if (value > max) { max = value; }
    }
    return max;`;
  }

  prime() {
    return `if (y * yHat * f32(2) < f32(1)) { return -y * yHat; } return 0;`;
  }
}
