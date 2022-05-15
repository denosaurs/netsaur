export interface CPUActivationFn {
  activate(val: number): number;
  prime(val: number): number;
}

export class Sigmoid implements CPUActivationFn {
  activate(val: number): number {
    return 1 / (1 + Math.exp(-val));
  }

  prime(val: number): number {
    return val * (1 - val);
  }
}

export class Tanh implements CPUActivationFn {
  activate(val: number): number {
    return Math.tanh(val);
  }

  prime(val: number): number {
    return 1 - (val * val);
  }
}

export class Relu implements CPUActivationFn {
  activate(val: number): number {
    return Math.max(0, val);
  }

  prime(val: number): number {
    return val > 0 ? 1 : 0;
  }
}

export class LeakyRelu implements CPUActivationFn {
  activate(val: number): number {
    return val > 0 ? val : 0.01 * val;
  }

  prime(val: number): number {
    return val > 0 ? 1 : 0.01;
  }
}

export class Elu implements CPUActivationFn {
  activate(val: number): number {
    return val >= 0 ? val : Math.exp(val) - 1;
  }

  prime(val: number): number {
    return val > 0 ? 1 : Math.exp(val);
  }
}
export class Linear implements CPUActivationFn {
  activate(val: number): number {
    return val;
  }

  prime(_val: number): number {
    return 1;
  }
}