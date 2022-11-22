import { Tensor } from "../../../core/tensor.ts";
import { CPUTensor, LayerJSON, Rank, Shape } from "../../../core/types.ts";
import { iterate1D } from "../../../core/util.ts";
import {
  relu,
  relu_prime,
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
  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    return input;
  }
  backPropagate(_: CPUTensor<Rank>) {}
  // deno-lint-ignore require-await
  async toJSON() {
    return { type: this.type };
  }
  static fromJSON(_: LayerJSON): ActivationCPULayer {
    return new this();
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

  backPropagate(error: CPUTensor<Rank>) {
    return error;
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
