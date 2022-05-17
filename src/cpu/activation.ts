export interface CPUActivationFn {
  activate(val: number): number;
  prime(val: number): number;
}

/**
 * Linear activation function f(x) = x
 */
export class Linear implements CPUActivationFn {
  activate(val: number): number {
    return val;
  }

  prime(_val: number): number {
    return 1;
  }
}

/**
 * Sigmoid activation function f(x) = 1 / (1 + e^(-x))
 */
export class Sigmoid implements CPUActivationFn {
  activate(val: number): number {
    return 1 / (1 + Math.exp(-val));
  }

  prime(val: number): number {
    return val * (1 - val);
  }
}

/**
 * Tanh activation function f(x) = (e^x - e^-x) / (e^x + e^-x)
 * This is the same as the sigmoid function, but is more robust to outliers
 */
export class Tanh implements CPUActivationFn {
  activate(val: number): number {
    return Math.tanh(val);
  }

  prime(val: number): number {
    return 1 - (val * val);
  }
}

/**
 * ReLU activation function f(x) = max(0, x)
 * This is a rectified linear unit, which is a smooth approximation to the sigmoid function.
 */
export class Relu implements CPUActivationFn {
  activate(val: number): number {
    return Math.max(0, val);
  }

  prime(val: number): number {
    return val > 0 ? 1 : 0;
  }
}

/**
 * Relu6 activation function f(x) = min(max(0, x), 6)
 * This is a rectified linear unit with a 6-value output range.
 */
export class Relu6 implements CPUActivationFn {
  activate(val: number): number {
    return Math.min(Math.max(0, val), 6);
  }

  prime(val: number): number {
    return val > 0 ? 1 : 0;
  }
}

/**
 * Leaky ReLU activation function f(x) = x if x > 0, 0.01 * x otherwise
 */
export class LeakyRelu implements CPUActivationFn {
  activate(val: number): number {
    return val > 0 ? val : 0.01 * val;
  }

  prime(val: number): number {
    return val > 0 ? 1 : 0.01;
  }
}

/**
 * Elu activation function f(x) = x if x >= 0, 1.01 * (e^x - 1) otherwise
 * This is a rectified linear unit with an exponential output range.
 */
export class Elu implements CPUActivationFn {
  activate(val: number): number {
    return val >= 0 ? val : Math.exp(val) - 1;
  }

  prime(val: number): number {
    return val > 0 ? 1 : Math.exp(val);
  }
}

/**
 * Selu activation function f(x) = x if x >= 0, 1.67 * (e^x - 1) otherwise
 * This is a scaled version of the Elu function, which is a smoother approximation to the ReLU function.
 */
export class Selu implements CPUActivationFn {
  activate(val: number): number {
    return val >= 0 ? val : 1.0507 * (Math.exp(val) - 1);
  }

  prime(val: number): number {
    return val > 0 ? 1 : 1.0507 * Math.exp(val);
  }
}
