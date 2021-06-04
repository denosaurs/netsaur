export interface Activation {
    activate(val: number): number;
    measure(weight: number, error: number): number;
}

export class Sigmoid {
  activate(val: number): number {
    return 1 / (1 + Math.exp(-val));
  }
  measure(weight: number, error: number): number {
    return weight * (1 - weight) * error;
  }
}

export class Tanh {
  activate(weight: number): number {
    return Math.tanh(weight);
  }
  measure(weight: number, error: number): number {
    return (1 - weight * weight) * error;
  }
}

export class Relu {
  activate(weight: number): number {
    return Math.max(0, weight);
  }
  measure(weight: number, delta: number): number {
    if (weight <= 0) {
      return 0;
    }
    return delta;
  }
}

export class LeakyRelu {
  activate(weight: number): number {
    return weight > 0 ? weight : 0.01 * weight;
  }
  measure(weight: number, error: number): number {
    return weight > 0 ? error : 0.01 * error;
  }
}
