import { Tensor } from "../../../core/tensor.ts";
import { CPUTensor, LayerJSON, Rank, Shape } from "../../../core/types.ts";
import { iterate1D, iterate3D } from "../../../core/util.ts";
import {
  elu,
  elu_prime,
  leakyrelu,
  leakyrelu_prime,
  relu,
  relu6,
  relu6_prime,
  relu_prime,
  selu,
  selu_prime,
  sigmoid,
  sigmoid_prime,
  tanh_prime,
} from "../kernels/mod.ts";

export class ActivationCPULayer {
  type!: string;
  output!: CPUTensor<Rank>;
  reset() {}
  initialize(shape: Shape[Rank]) {
    this.output = new Tensor(new Float32Array(), shape);
  }
  // deno-lint-ignore require-await
  async toJSON() {
    return { type: this.type };
  }
  static fromJSON(_: LayerJSON): ActivationCPULayer {
    return new this();
  }
}

/**
 * Elu Layer
 */
export class EluCPULayer extends ActivationCPULayer {
  type = "elu";
  input!: CPUTensor<Rank>;

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input;
    this.output = new Tensor(input.data.map(elu), input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>) {
    const dInput = new Float32Array(this.input.data.length);
    for (let i = 0; i < dInput.length; i++) {
      dInput[i] = dError.data[i] * elu_prime(this.input.data[i]);
    }
    return new Tensor(dInput, this.output.shape);
  }
}

/**
 * LeakyRelu Layer
 */
export class LeakyReluCPULayer extends ActivationCPULayer {
  type = "leakyrelu";
  input!: CPUTensor<Rank>;

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input;
    this.output = new Tensor(input.data.map(leakyrelu), input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>) {
    const dInput = new Float32Array(this.input.data.length);
    for (let i = 0; i < dInput.length; i++) {
      dInput[i] = dError.data[i] * leakyrelu_prime(this.input.data[i]);
    }
    return new Tensor(dInput, this.output.shape);
  }
}

/**
 * Softmax Layer
 */
export class SoftmaxCPULayer extends ActivationCPULayer {
  type = "softmax";

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    const output = new Float32Array(input.data.length);
    let sum = 0;
    iterate1D(input.data.length, (i) => {
      output[i] = Math.exp(input.data[i]);
      sum += output[i];
    });
    iterate1D(input.data.length, (i) => {
      output[i] /= sum;
    });
    this.output = new Tensor(output, input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>): CPUTensor<Rank> {
    const dInput = new Float32Array(this.output.data.length);
    const batches = this.output.shape.at(-1)!;
    const length = this.output.data.length / batches;
    iterate3D([batches, length, length], (z, x, y) => {
      const out1 = this.output.data[z * batches + x];
      const out2 = this.output.data[z * batches + y];
      const dActivation = x == y ? out1 * (1 - out1) : -out1 * out2;
      dInput[z * batches + x] += dActivation * dError.data[z * batches + y];
    });
    return new Tensor(dInput, this.output.shape);
  }
}

/**
 * Sigmoid Layer
 */
export class SigmoidCPULayer extends ActivationCPULayer {
  type = "sigmoid";
  input!: CPUTensor<Rank>;

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input;
    this.output = new Tensor(input.data.map(sigmoid), input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>) {
    const dInput = new Float32Array(this.input.data.length);
    for (let i = 0; i < dInput.length; i++) {
      dInput[i] = dError.data[i] * sigmoid_prime(this.output.data[i]);
    }
    return new Tensor(dInput, this.output.shape);
  }
}

/**
 * Tanh Layer
 */
export class TanhCPULayer extends ActivationCPULayer {
  type = "tanh";
  input!: CPUTensor<Rank>;

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input;
    this.output = new Tensor(input.data.map(Math.tanh), input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>) {
    const dInput = new Float32Array(this.input.data.length);
    for (let i = 0; i < dInput.length; i++) {
      dInput[i] = dError.data[i] * tanh_prime(this.output.data[i]);
    }
    return new Tensor(dInput, this.output.shape);
  }
}

/**
 * Relu Layer
 */
export class ReluCPULayer extends ActivationCPULayer {
  type = "relu";
  input!: CPUTensor<Rank>;

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input;
    this.output = new Tensor(input.data.map(relu), input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>) {
    const dInput = new Float32Array(this.input.data.length);
    for (let i = 0; i < dInput.length; i++) {
      dInput[i] = dError.data[i] * relu_prime(this.input.data[i]);
    }
    return new Tensor(dInput, this.output.shape);
  }
}

/**
 * Relu6 Layer
 */
export class Relu6CPULayer extends ActivationCPULayer {
  type = "relu6";
  input!: CPUTensor<Rank>;

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input;
    this.output = new Tensor(input.data.map(relu6), input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>) {
    const dInput = new Float32Array(this.input.data.length);
    for (let i = 0; i < dInput.length; i++) {
      dInput[i] = dError.data[i] * relu6_prime(this.input.data[i]);
    }
    return new Tensor(dInput, this.output.shape);
  }
}

/**
 * Selu Layer
 */
export class SeluCPULayer extends ActivationCPULayer {
  type = "selu";
  input!: CPUTensor<Rank>;

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    this.input = input;
    this.output = new Tensor(input.data.map(selu), input.shape);
    return this.output;
  }

  backPropagate(dError: CPUTensor<Rank>) {
    const dInput = new Float32Array(this.input.data.length);
    for (let i = 0; i < dInput.length; i++) {
      dInput[i] = dError.data[i] * selu_prime(this.input.data[i]);
    }
    return new Tensor(dInput, this.output.shape);
  }
}
