export interface GPUActivationFn {
  activate(type: string): string;
  prime(type: string): string;
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
export class Linear implements GPUActivationFn {
  activate(_type: string): string {
    return `return weighted_sum`;
  }

  prime(_type: string): string {
    return `return error`;
  }
}
