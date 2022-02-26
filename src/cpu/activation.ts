export interface CPUActivationFn {
  activate(val: number): number;
  prime(val: number): number;
}

export class Sigmoid implements CPUActivationFn {
  activate(val: number): number {
    return 1 / (1 + Math.exp(-val));
  }
  
  prime(val: number): number {
    return this.activate(val) * (1-this.activate(val))
  }
}

export class Tanh implements CPUActivationFn {
  activate(val: number): number {
    return Math.tanh(val);
  }

  prime(val: number): number {
    return 1 - Math.pow(this.activate(val), 2);
  }
}

export class Relu implements CPUActivationFn {
  activate(val: number): number {
    return Math.max(0, val);
  }

  prime(val: number): number {
    return val > 0 ? 1 : 0
  }
}

export class LeakyRelu implements CPUActivationFn {
  activate(val: number): number {
    return val > 0 ? val : 0.01 * val;
  }

  prime(val: number): number {
    return val > 0 ? 1 : 0.01
  }
}
