export interface GPUActivationFn {
  activate(type: string): string;
  prime(type: string): string;
}

export class Linear implements GPUActivationFn {
  activate(_: string): string {
    return `return weighted_sum`;
  }

  prime(_: string): string {
    return `return error`;
  }
}

export class Sigmoid implements GPUActivationFn {
  activate(type: string): string {
    return `return ${type}(1) / (${type}(1) + exp(-weighted_sum))`;
  }

  prime(type: string): string {
    return `return output * (${type}(1) - output)`;
  }
}

export class Tanh implements GPUActivationFn {
  activate(_: string): string {
    return `return tanh(weighted_sum)`;
  }

  prime(type: string): string {
    return `return ${type}(1) - output * output`;
  }
}

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

export class Selu implements GPUActivationFn {
  activate(type: string): string {
    return `return ${type}(weighted_sum) + ${type}(weighted_sum) * (1 - ${type}(weighted_sum)) * ${type}(1.67326)`;
  }

  prime(type: string): string {
    return `if (weighted_sum > ${type}(0)) {
            return error;
        }
        return ${type}(error) * (1 - ${type}(weighted_sum)) * ${type}(1.67326);`;
  }
}
