import { DataType, DataTypeArray } from "../deps.ts";
import { ConvLayer, DenseLayer, PoolLayer } from "../mod.ts";
import { ConvCPULayer } from "../backends/cpu/layers/conv.ts";
import { DenseCPULayer } from "../backends/cpu/layers/dense.ts";
import { DenseGPULayer } from "../backends/gpu/layers/dense.ts";
import { GPUMatrix } from "../backends/gpu/matrix.ts";
import { CPUMatrix } from "../backends/cpu/matrix.ts";
import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";

export interface LayerJSON {
  outputSize: number | Size2D;
  activationFn?: string;
  costFn?: string;
  type: string;
  input: MatrixJSON;
  weights?: MatrixJSON;
  biases?: MatrixJSON;
  output: MatrixJSON;
  error?: MatrixJSON;
  cost?: MatrixJSON;
  kernel?: MatrixJSON;
  padded?: MatrixJSON;
  stride?: number;
  padding?: number;
}

export interface NetworkJSON {
  costFn?: string;
  type: "NeuralNetwork";
  sizes: (number | Size2D)[];
  input: Size | undefined;
  layers: LayerJSON[];
  output: LayerJSON;
}

export interface MatrixJSON {
  // deno-lint-ignore no-explicit-any
  data: any;
  x: number;
  y: number;
  type?: DataType
}
export interface Backend<T extends DataType = DataType> {
  // deno-lint-ignore no-explicit-any
  addLayer(layer: Layer | any, index?: number): void;
  // getOutput(): DataTypeArray<T> | any;
  train(
    // deno-lint-ignore no-explicit-any
    datasets: DataSet[] | any,
    epochs: number,
    batches: number,
    learningRate: number,
  ): void;
  // deno-lint-ignore no-explicit-any
  predict(input: DataTypeArray<T> | any): DataTypeArray<T> | any;
  save(input: string): void;
  toJSON(): NetworkJSON | Promise<NetworkJSON>| undefined;
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
  activation?: Activation;
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

export type Cost = "crossentropy" | "hinge" | "mse";


export type Shape = number;
/**
 * NumberArray is a typed array of numbers.
 */
export type NumberArray<T extends DataType = DataType> =
  | DataTypeArray<T>
  | Array<number>
  // TODO: fix
  // deno-lint-ignore no-explicit-any
  | any;
/**
 * DataSet is a container for training data.
 */
export type DataSet = {
  inputs: NumberArray;
  outputs: NumberArray;
};
