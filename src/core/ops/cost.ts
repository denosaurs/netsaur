import { Engine } from "../engine.ts";

// deno-lint-ignore no-explicit-any
export function CrossEntropy(yHat: any, y: any, prime = false) {
  if (!Engine.kernels.crossentropy) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.crossentropy_prime(yHat, y)
    : Engine.kernels.crossentropy(yHat, y);
}

// deno-lint-ignore no-explicit-any
export function Hinge(yHat: any, y: any, prime = false) {
  if (!Engine.kernels.hinge) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.hinge_prime(yHat, y)
    : Engine.kernels.hinge(yHat, y);
}
