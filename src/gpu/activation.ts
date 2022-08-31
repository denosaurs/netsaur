export interface GPUActivationFn {
  activate(type: string): string;
  prime(type: string): string;
}

/**
 * Linear activation function f(x) = x
 */
export class Linear implements GPUActivationFn {
  activate(_: string): string {
    return `return weighted_sum`;
  }

  prime(type: string): string {
    return `return ${type}(1)`;
  }
}

/**
 * Sigmoid activation function f(x) = 1 / (1 + e^(-x))
 */
export class Sigmoid implements GPUActivationFn {
  activate(type: string): string {
    return `return ${type}(1) / (${type}(1) + exp(-weighted_sum))`;
  }

  prime(type: string): string {
    return `return output * (${type}(1) - output)`;
  }
}

/**
 * Tanh activation function f(x) = (e^x - e^-x) / (e^x + e^-x)
 * This is the same as the sigmoid function, but is more robust to outliers
 */
export class Tanh implements GPUActivationFn {
  activate(_: string): string {
    return `return tanh(weighted_sum)`;
  }

  prime(type: string): string {
    return `return ${type}(1) - output * output`;
  }
}

/**
 * ReLU activation function f(x) = max(0, x)
 * This is a rectified linear unit, which is a smooth approximation to the sigmoid function.
 */
export class Relu implements GPUActivationFn {
  activate(type: string): string {
    return `return max(${type}(0), weighted_sum)`;
  }

  prime(type: string): string {
    return `if (weighted_sum <= ${type}(0)) {
            return ${type}(0);
        }
        return errror;`;
  }
}

/**
 * Relu6 activation function f(x) = min(max(0, x), 6)
 * This is a rectified linear unit with a 6-value output range.
 */
export class Relu6 implements GPUActivationFn {
  activate(type: string): string {
    return `return min(max(${type}(0), weighted_sum), ${type}(6))`;
  }

  prime(type: string): string {
    return `if (weighted_sum <= ${type}(0)) {
            return ${type}(0);
        }
        if (weighted_sum >= ${type}(6)) {
            return ${type}(6);
        }
        return error;`;
  }
}

/**
 * Leaky ReLU activation function f(x) = x if x > 0, 0.01 * x otherwise
 */
export class LeakyRelu implements GPUActivationFn {
  activate(type: string): string {
    return `if (weighted_sum > ${type}(0)) {
            return weighted_sum;
        }
        return ${type}(f32(weighted_sum) * 0.01);`;
  }

  prime(type: string): string {
    return `if (weighted_sum > ${type}(0)) {
            return error;
        }
        return ${type}(f32(error) * 0.01);`;
  }
}

/**
 * Elu activation function f(x) = x if x >= 0, 1.01 * (e^x - 1) otherwise
 * This is a rectified linear unit with an exponential output range.
 */
export class Elu implements GPUActivationFn {
  activate(type: string): string {
    return `if (weighted_sum > ${type}(0)) {
            return weighted_sum;
        }
        return ${type}(exp(weighted_sum) - ${type}(1));`;
  }

  prime(type: string): string {
    return `if (weighted_sum > ${type}(0)) {
            return error;
        }
        return ${type}(exp(weighted_sum) - ${type}(1));`;
  }
}

/**
 * Selu activation function f(x) = x if x >= 0, 1.67 * (e^x - 1) otherwise
 * This is a scaled version of the Elu function, which is a smoother approximation to the ReLU function.
 */
export class Selu implements GPUActivationFn {
  activate(type: string): string {
    return `return ${type}(weighted_sum) + ${type}(weighted_sum) * (${type}(1) - ${type}(weighted_sum)) * ${type}(1.67326)`;
  }

  prime(type: string): string {
    return `if (weighted_sum > ${type}(0)) {
            return error;
        }
        return ${type}(error) * (${type}(1) - ${type}(weighted_sum)) * ${type}(1.67326);`;
  }
}
