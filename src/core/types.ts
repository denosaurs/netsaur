import { Tensor } from "./tensor/tensor.ts";
import { Rank, Shape } from "./api/shape.ts";
import { Layer } from "./api/layer.ts";
import { NetworkJSON } from "../model/types.ts";

/**
 * The Backend is responsible for eveything related to the neural network.
 */
export interface Backend {
  train(datasets: DataSet[], epochs: number, rate: number): void;

  predict(input: Tensor<Rank>): Promise<Tensor<Rank>>;

  save(input: string): void;

  toJSON(): Promise<NetworkJSON>;
}

/**
 * NetworkConfig represents the configuration of a neural network.
 */
export type NetworkConfig = {
  size: Shape[Rank];
  layers: Layer[];
  cost: Cost;
  silent?: boolean;
};

/**
 * Activation functions are used to transform the output of a layer into a new output.
 */
export enum Activation {
  /**
   * Sigmoid activation function f(x) = 1 / (1 + e^(-x))
   */
  Sigmoid = "sigmoid",

  /**
   * Tanh activation function f(x) = (e^x - e^-x) / (e^x + e^-x)
   * This is the same as the sigmoid function, but is more robust to outliers
   */
  Tanh = "tanh",

  /**
   * ReLU activation function f(x) = max(0, x)
   * This is a rectified linear unit, which is a smooth approximation to the sigmoid function.
   */
  Relu = "relu",

  /**
   * Relu6 activation function f(x) = min(max(0, x), 6)
   * This is a rectified linear unit with a 6-value output range.
   */
  Relu6 = "relu6",

  /**
   * Leaky ReLU activation function f(x) = x if x > 0, 0.01 * x otherwise
   */
  LeakyRelu = "leakyrelu",

  /**
   * Elu activation function f(x) = x if x >= 0, 1.01 * (e^x - 1) otherwise
   * This is a rectified linear unit with an exponential output range.
   */
  Elu = "elu",

  /**
   * Selu activation function f(x) = x if x >= 0, 1.67 * (e^x - 1) otherwise
   * This is a scaled version of the Elu function, which is a smoother approximation to the ReLU function.
   */
  Selu = "selu",

  /**
   * Linear activation function f(x) = x
   */
  Linear = "linear",
}

export enum Cost {
  /**
   * Cross entropy cost function is the standard cost function for binary classification.
   */
  CrossEntropy = "crossentropy",

  /**
   * Hinge cost function is the standard cost function for multi-class classification.
   */
  Hinge = "hinge",

  /**
   * Mean squared error cost function is the standard cost function for regression.
   */
  MSE = "mse",
}

/**
 * DataSet is a container for training data.
 */
export type DataSet = {
  inputs: Tensor<Rank>;
  outputs: Tensor<Rank>;
};

export enum LayerType {
  Dense = "dense",
  Activation = "activation",
  Conv2D = "conv2d",
  Dropout = "dropout",
  Pool = "pool",
  Flatten = "flatten",
  Softmax = "softmax",
}

/**
 * BackendType represents the type of backend to use.
 */
export enum BackendType {
  /**
   * CPU backend
   */
  CPU = "cpu",

  /**
   * GPU backend
   */
  GPU = "gpu",

  /**
   * Web Assembly backend
   */
  WASM = "wasm",
}

/**
 * Init represents the type of initialization to use.
 */
export enum Init {
  /**
   * Uniform initialization
   */
  Uniform = "uniform",

  /**
   * Xavier initialization
   */
  Xavier = "xavier",

  /**
   * XavierN initialization
   */
  XavierN = "xaviern",

  /**
   * Kaiming initialization
   */
  Kaiming = "kaiming",
}