import { Tensor, toData } from "./tensor.ts";
import { BackendType, Init, Rank, Shape } from "./types.ts";
import { iterate1D, Random } from "./util.ts";

export class Uniform {
  init<R extends Rank, B extends BackendType>(
    _input: Shape[Rank],
    weights: Shape[R],
    _?: Shape[Rank],
  ): Tensor<R, B> {
    const res = new Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.random(-1, 1));
    return new Tensor(toData(res), weights);
  }
}

export class Xavier {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    _?: Shape[Rank],
  ): Tensor<R, B> {
    const bounds = 1 / Math.sqrt(length(input));
    const res = new Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.random(-bounds, bounds));
    return new Tensor(toData(res), weights);
  }
}

export class XavierNorm {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    output: Shape[Rank],
  ): Tensor<R, B> {
    const bounds = Math.sqrt(6) / Math.sqrt(length(input) + length(output));
    const res = new Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.random(-bounds, bounds));
    return new Tensor(toData(res), weights);
  }
}

export class Kaiming {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    _?: Shape[Rank],
  ): Tensor<R, B> {
    const std = Math.sqrt(2 / length(input));
    const res = new Array(length(weights));
    iterate1D(res.length, (i) => res[i] = Random.gaussian(0, std));
    return new Tensor(toData(res), weights);
  }
}

function length(shape: Shape[Rank]) {
  let length = 1;
  shape.forEach((i) => length *= i);
  return length;
}

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
