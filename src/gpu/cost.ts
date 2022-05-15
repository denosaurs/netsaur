// deno-lint-ignore no-unused-vars
import { WebGPUData } from "../../deps.ts";

export interface GPUCostFunction {
  /** Return the cost associated with an output `a` and desired output `y`. */
  cost(type: string): string;

  /** Return the error delta from the output layer. */
  prime(type: string): string;
}

export class CrossEntropy implements GPUCostFunction {
  cost(_type: string) {
    return ``;
  }

  prime(_type: string) {
    return `return yHat - y`;
  }
}

export class Hinge implements GPUCostFunction {
  cost(_type: string) {
    return ``;
  }

  prime(_type: string) {
    return ``;
  }
}
