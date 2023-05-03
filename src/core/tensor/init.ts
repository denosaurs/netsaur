import { Tensor } from "./tensor.ts";
import { BackendType, Init } from "../types.ts";
import { Random } from "./random.ts";
import { iterate1D, length } from "./util.ts";
import { Rank, Shape } from "../api/shape.ts";

/**
 * setInit sets the initialization function.
 */
export function setInit(init: Init) {
  switch (init) {
    case "uniform":
      return new Uniform();
    case "xavier":
      return new Xavier();
    case "xaviern":
      return new XavierNorm();
    case "kaiming":
      return new Kaiming();
  }
}

/**
 * Uniform initializer
 */
export class Uniform {
  init<R extends Rank, B extends BackendType>(
    _input: Shape[Rank],
    weights: Shape[R],
    _?: Shape[Rank],
  ): Tensor<R, B> {
    const res = new Float32Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.random(-1, 1));
    return Tensor.from(res, weights);
  }
}

/**
 * Xavier initializer
 */
export class Xavier {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    _?: Shape[Rank],
  ): Tensor<R, B> {
    const bounds = 1 / Math.sqrt(length(input));
    const res = new Float32Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.random(-bounds, bounds));
    return Tensor.from(res, weights);
  }
}

/**
 * XavierNorm initializer
 */
export class XavierNorm {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    output: Shape[Rank],
  ): Tensor<R, B> {
    const bounds = Math.sqrt(6) / Math.sqrt(length(input) + length(output));
    const res = new Float32Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.random(-bounds, bounds));
    return Tensor.from(res, weights);
  }
}

/**
 * Kaiming initializer
 */
export class Kaiming {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    _?: Shape[Rank],
  ): Tensor<R, B> {
    const std = Math.sqrt(2 / length(input));
    const res = new Float32Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.gaussian(0, std));
    return Tensor.from(res, weights);
  }
}
