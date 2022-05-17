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
  cost(_type: string) {
    return ``;
  }

  prime(_type: string) {
    return `return yHat - y`;
  }
}

/**
 * Hinge cost function is the standard cost function for multiclass classification.
 */
export class Hinge implements GPUCostFunction {
  cost(_type: string) {
    return ``;
  }

  prime(type: string) {
    return `if (y * yHat * ${type}(2) < ${type}(1)) { return -y * yHat; } return 0;`;
  }
}
