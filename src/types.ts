import { DataType, DataTypeArray } from "../deps.ts";
import { ConvLayer, DenseLayer, PoolLayer } from "./mod.ts";
import { ConvCPULayer } from "./cpu/layers/conv.ts";
import { DenseCPULayer } from "./cpu/layers/dense.ts";
import { DenseGPULayer } from "./gpu/layers/dense.ts";
import { CPUActivationFn } from "./cpu/activation.ts";
import { GPUActivationFn } from "./gpu/activation.ts";
import { GPUMatrix } from "./gpu/matrix.ts";
import { CPUMatrix } from "./cpu/matrix.ts";
import { PoolCPULayer } from "./cpu/layers/pool.ts";

export interface LayerJSON {
  outputSize: number | Size2D;
  activation?: CPUActivationFn | GPUActivationFn;
  type: string;
}

export interface NetworkJSON {
  type: "NeuralNetwork";
  sizes: (number | Size2D)[];
  input: Size | undefined;
  layers: LayerJSON[];
  output: LayerJSON;
}

export interface Network<T extends DataType = DataType> {
  addLayer(layer: Layer): void;
  // deno-lint-ignore no-explicit-any
  getOutput(): DataTypeArray<T> | any;
  train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    learningRate: number,
  ): void;
  // deno-lint-ignore no-explicit-any
  predict(input: DataTypeArray<T>): DataTypeArray<T> | any;
  toJSON(): NetworkJSON;
  // deno-lint-ignore no-explicit-any
  getWeights(): (GPUMatrix | CPUMatrix | any)[];
  // deno-lint-ignore no-explicit-any
  getBiases(): (GPUMatrix | CPUMatrix | any)[];
}

/**
 * NetworkConfig represents the configuration of a neural network.
 */
export interface NetworkConfig {
  input?: Size;
  layers: Layer[];
  cost: Cost;
  silent?: boolean;
}

export type Layer = DenseLayer | ConvLayer | PoolLayer;

export type CPULayer = ConvCPULayer | DenseCPULayer | PoolCPULayer;

export type GPULayer = DenseGPULayer;

export interface DenseLayerConfig {
  size: Size;
  activation: Activation;
}

export interface ConvLayerConfig {
  activation: Activation;
  kernel: DataTypeArray;
  kernelSize: Size2D;
  padding?: number;
  stride?: number;
}

export interface PoolLayerConfig {
  stride: number;
}

export type Size = number | Size2D;

export type Size2D = { x: number; y: number };

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
