import { DataType, DataTypeArray } from "../deps.ts";

export interface Network<T extends DataType = DataType> {
  addLayers(layer: LayerConfig[]): void;
  getOutput(): DataTypeArray<T>;
  train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    learningRate: number,
  ): void;
  // deno-lint-ignore no-explicit-any
  predict(input: DataTypeArray<T>): DataTypeArray<T> | any;
}

/**
 * NetworkConfig represents the configuration of a neural network.
 */
export interface NetworkConfig {
  input?: InputConfig;
  hidden: LayerConfig[];
  cost: Cost;
  output: LayerConfig;
  silent?: boolean;
}
/**
 * LayerConfig is the configuration for a layer.
 */
export interface LayerConfig {
  size: number;
  activation: Activation;
}

/**
 * Activation functions are used to transform the output of a layer into a new output.
 */
export type Activation =
  | "sigmoid"
  | "tanh"
  | "relu"
  | "relu6"
  | "leakyrelu"
  | "elu"
  | "linear"
  | "selu";

export type Cost = "crossentropy" | "hinge";

export type Shape = number;
/**
 * InputConfig represents the configuration of the input layer.
 */
export type InputConfig = {
  size?: number;
  type: DataType;
};
/**
 * NumberArray is a typed array of numbers.
 */
export type NumberArray<T extends DataType = DataType> =
  | DataTypeArray<T>
  | Array<number>;
/**
 * DataSet is a container for training data.
 */
export type DataSet = {
  inputs: NumberArray;
  outputs: NumberArray;
};
