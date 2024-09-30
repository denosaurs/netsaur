import type { Tensor } from "./tensor/tensor.ts";
import type { Rank, Shape } from "./api/shape.ts";
import type { Layer } from "./api/layer.ts";
import type { Optimizer } from "./api/optimizer.ts";
import type { Scheduler } from "./api/scheduler.ts";
import type { PostProcessor } from "./api/postprocess.ts";

/**
 * The Backend is responsible for eveything related to the neural network.
 */
export interface Backend {
  /**
   * The train method is a function that trains a neural network using a set of training data.
   * It takes in an array of DataSet objects, the number of epochs to train for, and the learning rate.
   * The method modifies the weights and biases of the network to minimize the cost function and improve its accuracy on the training data.
   *
   * ```ts
   * net.train(datasets, 100, 10, 0.01);
   * ```
   */
  train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    rate: number,
  ): void;

  /**
   * The predict method is a function that takes in a Tensor object
   * representing the input to the neural network and returns a Promise that resolves to a Tensor object representing the output of the network.
   * This method is used to make predictions on new data after the network has been trained.
   *
   * ```ts
   * const input = new Tensor([1, 2, 3, 4]);
   * const output = await net.predict(input);
   * ```
   */
  predict(
    input: Tensor<Rank>,
    config: {postProcess: PostProcessor, outputShape?: Shape<Rank>},
    layers?: number[],
  ): Promise<Tensor<Rank>>;

  /**
   * The save method is a function that saves the network to a Uint8Array.
   * This method is used to save the network after it has been trained.
   *
   * ```ts
   * const buffer = net.save();
   * ```
   */
  save(): Uint8Array;

  /**
   * The saveFile method is a function that takes in a string representing the path to a file and saves the network to that file.
   * This method is used to save the network after it has been trained.
   *
   * ```ts
   * net.saveFile("model.bin");
   * ```
   */
  saveFile(path: string): void;
}

/**
 * NetworkConfig represents the configuration of a neural network.
 */
export interface NetworkConfig {
  /**
   * Input size of the neural network.
   */
  size: Shape<Rank>;

  /**
   * List of layers in the neural network.
   */
  layers: Layer[];

  /**
   * Cost function used to train the neural network.
   */
  cost?: Cost;

  /**
   * Optimizer to update parameters.
   */
  optimizer?: Optimizer;

  /**
   * Learning rate scheduler.
   */
  scheduler?: Scheduler;

  /**
   * Whether or not to silence the verbose messages.
   */
  silent?: boolean;

  /**
   * Minimum threshold for weight updates in each epoch.
   */
  tolerance?: number;

  /**
   * Number of disappointing iterations to allow before early stopping
   */
  patience?: number;
}

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
   * Gelu is the Gaussian Error Linear Unit
   */
  Gelu = "gelu",
}

export enum Cost {
  /**
   * Cross entropy cost function is the standard cost function for binary classification.
   */
  BinCrossEntropy = "bincrossentropy",

  /**
   * Cross entropy cost function is the standard cost function for classification.
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

  /**
   * Mean absolute error cost function is a popular cost function for regression.
   */
  MAE = "mae",

  /**
   * Huber is a cost function for regression and is less sensitive to outliers than the
   * squared error loss
   */
  Huber = "huber",

  /**
   * Smoothed hinge is a variant of the Huber cost function used for binary classification.
   * It is a smoothed version of hinge and is more robust to outliers.
   */
  SmoothHinge = "smoothhinge",

  /**
   * Tukey's biweight loss is a robust cost function for regression problems.
   */
  Tukey = "tukey",
}

export enum OptimizerType {
  SGD = "sgd",
  Adam = "adam",
  Nadam = "nadam",
  RMSProp = "rmsprop",
}

export enum SchedulerType {
  None = "none",
  LinearDecay = "lineardecay",
  ExponentialDecay = "exponentialdecay",
  OneCycle = "onecycle",
}

/**
 * DataSet is a container for training data.
 */
export interface DataSet {
  inputs: Tensor<Rank>;
  outputs: Tensor<Rank>;
}

export enum LayerType {
  Activation = "activation",
  BatchNorm1D = "batchnorm1d",
  BatchNorm2D = "batchnorm2d",
  Conv1D = "conv1d",
  Conv2D = "conv2d",
  ConvTranspose1D = "convtranspose1d",
  ConvTranspose2D = "convtranspose2d",
  Dense = "dense",
  Dropout1D = "dropout1d",
  Dropout2D = "dropout2d",
  Pool2D = "pool2d",
  Embedding = "embedding",
  Flatten = "flatten",
  LSTM = "lstm",
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

export type Padding = "same" | "valid"