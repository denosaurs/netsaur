import { BackendType } from "../types.ts";
import { Rank, Shape, Shape2D } from "./shape.ts";

/**
 * Incompatible Rank Error is thrown when a tensor is incompatible with a layer.
 */
export class IncompatibleRankError extends Error {
  constructor(shape: number, expected: number) {
    super(
      `Layer of rank ${expected} is incompatible with tensor of rank ${shape}`,
    );
  }
}

/**
 * Invalid Flatten Error is thrown when a tensor cannot be flattened.
 */
export class InvalidFlattenError extends Error {
  constructor(input: Shape<Rank>, output: Shape<Rank>) {
    super(`Cannot flatten tensor of shape ${input} to shape ${output}`);
  }
}

/**
 * No backend error is thrown when a backend is not initialized.
 */
export class NoBackendError extends Error {
  constructor(type: BackendType) {
    super(
      `${type} backend not initialized. Help: Did you forget to call setupBackend()?`,
    );
  }
}

/**
 * Invalid Pool Error is thrown when a tensor cannot be pooled.
 */
export class InvalidPoolError extends Error {
  constructor(size: Shape<Rank>, stride: Shape2D) {
    super(`Cannot pool shape ${size} with stride ${stride}`);
  }
}

/**
 * Invalid Activation Error is thrown when an activation function is invalid.
 */
export class ActivationError extends Error {
  constructor(activation: string) {
    super(`Unknown activation function: ${activation}.`);
  }
}
