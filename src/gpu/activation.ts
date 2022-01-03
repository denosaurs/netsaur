export interface GPUActivationFn {
    activate(type: string): string;
    measure(type: string): string;
}

export class Sigmoid implements GPUActivationFn {
    activate(type: string): string {
        return `return 1${type} / (1${type} + exp(-weighted_sum))`;
    }

    measure(type: string): string {
        return `return weight * (1${type} - weight) * error`;
    }
}

export class Tanh implements GPUActivationFn {
    activate(_: string): string {
        return `return tanh(weighted_sum)`;
    }

    measure(type: string): string {
        return `return (1${type} - weighted_sum * weighted_sum) * error`;
    }
}

export class Relu implements GPUActivationFn {
    activate(type: string): string {
        return `return max(0${type}, weighted_sum)`;
    }

    measure(type: string): string {
        return `if (weighted_sum <= 0${type}) {
            return 0${type};
        }
        return errror;`
    }
}

export class LeakyRelu implements GPUActivationFn {
    activate(type: string): string {
        return `if (weighted_sum > 0${type}) {
            return weighted_sum;
        }
        return ${type}(f32(weighted_sum) * 0.01);`;
    }

    measure(type: string): string {
        return `if (weighted_sum > 0${type}) {
            return error;
        }
        return ${type}(f32(error) * 0.01);`;
    }
}
