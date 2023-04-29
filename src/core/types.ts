import { Tensor } from "./tensor/tensor.ts";
import { Rank, Shape } from "./api/shape.ts";
import { Layer } from "./api/layer.ts";
import { NetworkJSON } from "../model/types.ts";
import { Data } from "../model/data/data.ts";

export interface Backend {
  initialize(inputSize: Shape[Rank], batches: number): Promise<void>;

  addLayer(layer: Layer): void;

  train(
    datasets: DataSet[] | Data,
    epochs: number,
    batches: number,
    learningRate: number,
  ): void;

  predict(input: Tensor<Rank, BackendType>): Tensor<Rank, BackendType>;

  save(input: string): void;
  
  toJSON(): Promise<NetworkJSON>;
}

/**
 * NetworkConfig represents the configuration of a neural network.
 */
export type NetworkConfig = {
  input?: Shape[Rank];
  layers: Layer[];
  cost: Cost;
  silent?: boolean;
};

/**
 * Activation functions are used to transform the output of a layer into a new output.
 */
export enum Activation {
  Sigmoid = "sigmoid",
  Tanh = "tanh",
  Relu = "relu",
  Relu6 = "relu6",
  LeakyRelu = "leakyrelu",
  Elu = "elu",
  Linear = "linear",
  Selu = "selu",
}

export enum Cost {
  CrossEntropy = "crossentropy",
  Hinge = "hinge",
  MSE = "mse",
}

/**
 * DataSet is a container for training data.
 */
export type DataSet = {
  inputs: Tensor<Rank, BackendType>;
  outputs: Tensor<Rank, BackendType>;
};

export enum LayerType {
  Dense = "dense",
  Activation  = "activation",
  Conv = "conv",
  Pool = "pool",
  Flatten = "flatten",
  Softmax = "softmax",
}

export enum BackendType {
  CPU = "cpu",
  GPU = "gpu",
}

export type Init = "uniform" | "xavier" | "xaviern" | "kaiming";

export interface InitFn {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    outputs: Shape[Rank],
  ): Tensor<R, B>;
}
