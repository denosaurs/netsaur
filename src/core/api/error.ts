import { Rank, Shape, Shape2D } from "./shape.ts";

export class IncompatibleRankError extends Error {
  constructor(shape: number, expected: number) {
    super(
      `Layer of rank ${expected} is incompatible with tensor of rank ${shape}`,
    );
  }
}

export class InvalidFlattenError extends Error {
  constructor(input: Shape[Rank], output: Shape[Rank]) {
    super(`Cannot flatten tensor of shape ${input} to shape ${output}`);
  }
}

export class NoWebGPUBackendError extends Error {
  constructor() {
    super(
      `WebGPU backend not initialized. Help: Did you forget to call setupBackend()?`,
    );
  }
}

export class InvalidPoolError extends Error {
  constructor(size: Shape[Rank], stride: Shape2D) {
    super(`Cannot pool shape ${size} with stride ${stride}`);
  }
}

export class ActivationError extends Error {
  constructor(activation: string) {
    super(`Unknown activation function: ${activation}.`);
  }
}
