import { Activation } from "../../core/types.ts";
import { ActivationError } from "../../core/util.ts";

export interface GPUActivationFn {
  name: string;
  activate(...values: string[]): string;
  prime(...values: string[]): string;
}

/**
 * Linear activation function f(x) = x
 */
export class Linear implements GPUActivationFn {
  name = "linear";
  activate() {
    return `return weighted_sum`;
  }

  prime() {
    return `return f32(1)`;
  }
}

/**
 * Sigmoid activation function f(x) = 1 / (1 + e^(-x))
 */
export class Sigmoid implements GPUActivationFn {
  name = "sigmoid";
  activate() {
    return `return f32(1) / (f32(1) + exp(-weighted_sum))`;
  }

  prime() {
    return `return output * (f32(1) - output)`;
  }
}

/**
 * Tanh activation function f(x) = (e^x - e^-x) / (e^x + e^-x)
 * This is the same as the sigmoid function, but is more robust to outliers
 */
export class Tanh implements GPUActivationFn {
  name = "tanh";
  activate() {
    return `return tanh(weighted_sum)`;
  }

  prime() {
    return `return f32(1) - output * output`;
  }
}

/**
 * ReLU activation function f(x) = max(0, x)
 * This is a rectified linear unit, which is a smooth approximation to the sigmoid function.
 */
export class Relu implements GPUActivationFn {
  name = "relu";
  activate() {
    return `return max(f32(0), weighted_sum)`;
  }

  prime() {
    return `if (weighted_sum <= f32(0)) {
            return f32(0);
        }
        return errror;`;
  }
}

/**
 * Relu6 activation function f(x) = min(max(0, x), 6)
 * This is a rectified linear unit with a 6-value output range.
 */
export class Relu6 implements GPUActivationFn {
  name = "relu6";
  activate() {
    return `return min(max(f32(0), weighted_sum), f32(6))`;
  }

  prime() {
    return `if (weighted_sum <= f32(0)) {
            return f32(0);
        }
        if (weighted_sum >= f32(6)) {
            return f32(6);
        }
        return error;`;
  }
}

/**
 * Leaky ReLU activation function f(x) = x if x > 0, 0.01 * x otherwise
 */
export class LeakyRelu implements GPUActivationFn {
  name = "leakyrelu";
  activate() {
    return `if (weighted_sum > f32(0)) {
            return weighted_sum;
        }
        return f32(f32(weighted_sum) * 0.01);`;
  }

  prime() {
    return `if (weighted_sum > f32(0)) {
            return error;
        }
        return f32(f32(error) * 0.01);`;
  }
}

/**
 * Elu activation function f(x) = x if x >= 0, 1.01 * (e^x - 1) otherwise
 * This is a rectified linear unit with an exponential output range.
 */
export class Elu implements GPUActivationFn {
  name = "elu";
  activate() {
    return `if (weighted_sum > f32(0)) {
            return weighted_sum;
        }
        return f32(exp(weighted_sum) - f32(1));`;
  }

  prime() {
    return `if (weighted_sum > f32(0)) {
            return error;
        }
        return f32(exp(weighted_sum) - f32(1));`;
  }
}

/**
 * Selu activation function f(x) = x if x >= 0, 1.67 * (e^x - 1) otherwise
 * This is a scaled version of the Elu function, which is a smoother approximation to the ReLU function.
 */
export class Selu implements GPUActivationFn {
  name = "selu";
  activate() {
    return `return f32(weighted_sum) + f32(weighted_sum) * (f32(1) - f32(weighted_sum)) * f32(1.67326)`;
  }

  prime() {
    return `if (weighted_sum > f32(0)) {
            return error;
        }
        return f32(error) * (f32(1) - f32(weighted_sum)) * f32(1.67326);`;
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
