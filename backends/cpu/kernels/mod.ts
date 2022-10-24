import { iterate1D } from "../../../core/util.ts";
import { DataTypeArray } from "../../../deps.ts";

export function linear(val: number): number {
  return val;
}

export function linear_prime(_val: number, error = 1): number {
  return error;
}

export function sigmoid(val: number): number {
  return 1 / (1 + Math.exp(-val));
}

export function sigmoid_prime(val: number, error = 1): number {
  return val * (1 - val) * error;
}

export function tanh(val: number): number {
  return Math.tanh(val);
}

export function tanh_prime(val: number, error = 1): number {
  return (1 - (val * val)) * error;
}

export function relu(val: number) {
  return Math.max(0, val);
}

export function relu_prime(val: number, error = 1) {
  return (val > 0 ? error : 0);
}

export function relu6(val: number) {
  return Math.min(Math.max(0, val), 6);
}

export function relu6_prime(val: number, error = 1) {
  return (val > 0 ? error : 0);
}

export function leakyrelu(val: number): number {
  return val > 0 ? val : 0.01 * val;
}

export function leakyrelu_prime(val: number, error = 1): number {
  return val > 0 ? error : 0.01;
}

export function elu(val: number): number {
  return val >= 0 ? val : Math.exp(val) - 1;
}

export function elu_prime(val: number, error = 1): number {
  return val > 0 ? error : Math.exp(val);
}

export function selu(val: number): number {
  return val >= 0 ? val : 1.0507 * (Math.exp(val) - 1);
}

export function selu_prime(val: number, error = 1): number {
  return val > 0 ? error : 1.0507 * Math.exp(val);
}

export function crossentropy(yHat: DataTypeArray, y: DataTypeArray) {
  let sum = 0;
  iterate1D(yHat.length, (i: number) => {
    sum += -y * Math.log(yHat[i]) - (1 - y[i]) * Math.log(1 - yHat[i]);
  });
  return sum;
}

export function crossentropy_prime(yHat: number, y: number) {
  return yHat - y;
}

export function hinge(yHat: DataTypeArray, y: DataTypeArray) {
  let max = -Infinity;
  iterate1D(yHat.length, (i: number) => {
    const value = y[i] - (1 - 2 * y[i]) * yHat[i];
    if (value > max) max = value;
  });
  return max;
}

export function hinge_prime(yHat: number, y: number) {
  return y * yHat * 2 < 1 ? -y * yHat : 0;
}
