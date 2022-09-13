import { DataType, DataTypeArray } from "../deps.ts";
import { DenseLayer,ConvLayer } from "./mod.ts";
import { ConvCPULayer } from "./cpu/layers/conv.ts";
import { DenseCPULayer } from "./cpu/layers/dense.ts";
import { DenseGPULayer } from "./gpu/layers/dense.ts";

export interface Network<T extends DataType = DataType> {
  addLayers(layer: Layer[]): void;
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
  hidden: Layer[];
  cost: Cost;
  output: Layer;
  silent?: boolean;
}

export type Layer = DenseLayer | ConvLayer;

export type CPULayer = ConvCPULayer | DenseCPULayer;

export type GPULayer = DenseGPULayer;

export interface DenseLayerConfig {
  size: Size;
  activation: Activation;
}

export interface ConvLayerConfig {
  size: Size;
  activation: Activation;
  padding?: number;
  stride?: number;
}

export type Size = number | Size2D

export type Size2D = {x: number, y: number}

export type Backend = "gpu" | "cpu" | "GPU" | "CPU";
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
