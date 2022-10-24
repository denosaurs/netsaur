import { Engine } from "./engine.ts";

export function Sigmoid(data: number, prime = false, error = 1) {
  if (!Engine.ops.sigmoid) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.ops.sigmoid_prime(data, error)
    : Engine.ops.sigmoid(data);
}

export function Tanh(data: number, prime = false, error = 1) {
  if (!Engine.ops.tanh) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime ? Engine.ops.tanh_prime(data, error) : Engine.ops.tanh(data);
}

export function Relu(data: number, prime = false, error = 1) {
  if (!Engine.ops.relu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime ? Engine.ops.relu_prime(data, error) : Engine.ops.relu(data);
}

export function Relu6(data: number, prime = false, error = 1) {
  if (!Engine.ops.relu6) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime ? Engine.ops.relu6_prime(data, error) : Engine.ops.relu6(data);
}

export function LeakyRelu(data: number, prime = false, error = 1) {
  if (!Engine.ops.leakyrelu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime
    ? Engine.ops.leakyrelu_prime(data, error)
    : Engine.ops.leakyrelu(data);
}

export function Elu(data: number, prime = false, error = 1) {
  if (!Engine.ops.elu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime ? Engine.ops.elu_prime(data, error) : Engine.ops.elu(data);
}

export function Selu(data: number, prime = false, error = 1) {
  if (!Engine.ops.selu) {
    throw new Error("Current backend doesn't support this feature yet");
  }
  return prime ? Engine.ops.selu_prime(data, error) : Engine.ops.selu(data);
}
