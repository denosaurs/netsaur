import { Activation } from "../../core/types.ts";
import { ActivationError } from "../../core/util.ts";

export interface CPUActivationFn {
  name: string;
  activate(val: number): number;
  prime(val: number, error?: number): number;
}

/**
 * Linear activation function f(x) = x
 */
export class Linear implements CPUActivationFn {
  name = "linear";
  activate(val: number): number {
    return val;
  }

  prime(_val: number, error = 1): number {
    return error;
  }
}

/**
 * Sigmoid activation function f(x) = 1 / (1 + e^(-x))
 */
export class Sigmoid implements CPUActivationFn {
  name = "sigmoid";
  activate(val: number): number {
    return 1 / (1 + Math.exp(-val));
  }

  prime(val: number, error = 1): number {
    return val * (1 - val) * error;
  }
}

/**
 * Tanh activation function f(x) = (e^x - e^-x) / (e^x + e^-x)
 * This is the same as the sigmoid function, but is more robust to outliers
 */
export class Tanh implements CPUActivationFn {
  name = "tanh";
  activate(val: number): number {
    return Math.tanh(val);
  }

  prime(val: number, error = 1): number {
    return (1 - (val * val)) * error;
  }
}

/**
 * ReLU activation function f(x) = max(0, x)
 * This is a rectified linear unit, which is a smooth approximation to the sigmoid function.
 */
export class Relu implements CPUActivationFn {
  name = "relu";
  activate(val: number): number {
    return Math.max(0, val);
  }

  prime(val: number, error = 1): number {
    return (val > 0 ? error : 0);
  }
}

/**
 * Relu6 activation function f(x) = min(max(0, x), 6)
 * This is a rectified linear unit with a 6-value output range.
 */
export class Relu6 implements CPUActivationFn {
  name = "relu6";
  activate(val: number): number {
    return Math.min(Math.max(0, val), 6);
  }

  prime(val: number, error = 1): number {
    return (val > 0 ? error : 0);
  }
}

/**
 * Leaky ReLU activation function f(x) = x if x > 0, 0.01 * x otherwise
 */
export class LeakyRelu implements CPUActivationFn {
  name = "leakyrelu";
  activate(val: number): number {
    return val > 0 ? val : 0.01 * val;
  }

  prime(val: number, error = 1): number {
    return val > 0 ? error : 0.01;
  }
}

/**
 * Elu activation function f(x) = x if x >= 0, 1.01 * (e^x - 1) otherwise
 * This is a rectified linear unit with an exponential output range.
 */
export class Elu implements CPUActivationFn {
  name = "elu";
  activate(val: number): number {
    return val >= 0 ? val : Math.exp(val) - 1;
  }

  prime(val: number, error = 1): number {
    return val > 0 ? error : Math.exp(val);
  }
}

/**
 * Selu activation function f(x) = x if x >= 0, 1.67 * (e^x - 1) otherwise
 * This is a scaled version of the Elu function, which is a smoother approximation to the ReLU function.
 */
export class Selu implements CPUActivationFn {
  name = "selu";
  activate(val: number): number {
    return val >= 0 ? val : 1.0507 * (Math.exp(val) - 1);
  }

  prime(val: number, error = 1): number {
    return val > 0 ? error : 1.0507 * Math.exp(val);
  }
}

export function setActivation(activation: Activation) {
  switch (activation) {
    case "sigmoid":
      return new Sigmoid();
    case "leakyrelu":
      return new LeakyRelu();
    case "tanh":
      return new Tanh();
    case "relu":
      return new Relu();
    case "relu6":
      return new Relu6();
    case "elu":
      return new Elu();
    case "selu":
      return new Selu();
    case "linear":
      return new Linear();
    default:
      throw new ActivationError(activation);
  }
}