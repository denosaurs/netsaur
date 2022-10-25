import { Engine } from "../engine.ts";

export function Sigmoid(data: number, prime = false, error = 1) {
  if (!Engine.kernels.sigmoid) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.sigmoid_prime(data, error)
    : Engine.kernels.sigmoid(data);
}

export function Tanh(data: number, prime = false, error = 1) {
  if (!Engine.kernels.tanh) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.tanh_prime(data, error)
    : Engine.kernels.tanh(data);
}

export function Relu(data: number, prime = false, error = 1) {
  if (!Engine.kernels.relu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.relu_prime(data, error)
    : Engine.kernels.relu(data);
}

export function Relu6(data: number, prime = false, error = 1) {
  if (!Engine.kernels.relu6) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.relu6_prime(data, error)
    : Engine.kernels.relu6(data);
}

export function LeakyRelu(data: number, prime = false, error = 1) {
  if (!Engine.kernels.leakyrelu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.leakyrelu_prime(data, error)
    : Engine.kernels.leakyrelu(data);
}

export function Elu(data: number, prime = false, error = 1) {
  if (!Engine.kernels.elu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.elu_prime(data, error)
    : Engine.kernels.elu(data);
}

export function Selu(data: number, prime = false, error = 1) {
  if (!Engine.kernels.selu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.kernels.selu_prime(data, error)
    : Engine.kernels.selu(data);
}
