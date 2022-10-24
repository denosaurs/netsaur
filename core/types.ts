import { DataType, DataTypeArray } from "../deps.ts";
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
  strides?: Size;
  padding?: number;
  mode?: "max" | "avg";
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
  type?: DataType;
}

export interface Backend<T extends DataType = DataType> {
  // deno-lint-ignore no-explicit-any
  layers: Array<any>;
  initialize(
    inputSize: Size,
    batches: number,
    type?: DataType,
  ): void | Promise<void>;
  // deno-lint-ignore no-explicit-any
  addLayer(layer: Layer | any, index?: number): void;
  train(
    // deno-lint-ignore no-explicit-any
    datasets: DataSet[] | any,
    epochs: number,
    batches: number,
    learningRate: number,
  ): void;
  // deno-lint-ignore no-explicit-any
  predict(input: DataTypeArray<T> | any): DataTypeArray<T> | any;
  // deno-lint-ignore no-explicit-any
  feedForward(input: any): Promise<any> | any;
  save(input: string): void;
  toJSON(): NetworkJSON | Promise<NetworkJSON> | undefined;
  // deno-lint-ignore no-explicit-any
  getWeights(): (GPUMatrix | CPUMatrix | any)[];
  // deno-lint-ignore no-explicit-any
  getBiases(): (GPUMatrix | CPUMatrix | any)[];
}

export interface TensorBackend {
  tensor2D(values: TensorLike, width: number, height: number): Tensor2D;
  tensor1D(values: TensorLike): Tensor1D;
}

export type Tensor2DCPU = CPUMatrix;
export type Tensor2DGPU = GPUMatrix;
// deno-lint-ignore no-explicit-any
export type Tensor2DNative = any;
export type Tensor2D = Tensor2DCPU | Tensor2DGPU | Tensor2DNative;
// deno-lint-ignore no-explicit-any
export type Tensor1D = Float32Array | any;

/**
 * NetworkConfig represents the configuration of a neural network.
 */
export interface NetworkConfig {
  input?: Size;
  layers: Layer[];
  cost: Cost;
  silent?: boolean;
}

// deno-lint-ignore no-explicit-any
export type Layer = any;

export type CPULayer = ConvCPULayer | DenseCPULayer | PoolCPULayer;

export type GPULayer = DenseGPULayer;

export interface DenseLayerConfig {
  size: Size;
  activation?: Activation;
}

export interface ConvLayerConfig {
  activation?: Activation;
  kernel?: DataTypeArray;
  kernelSize: Size2D;
  padding?: number;
  unbiased?: boolean;
  strides?: Size;
}

export interface PoolLayerConfig {
  strides?: Size;
  mode?: "max" | "avg";
}

export type Size = number | Size2D;

export type Size1D = number | { x: number };

export type Size2D = { x: number; y: number };

export type Size3D = { x: number; y: number; z: number };

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

export type ArrayMap =
  | number
  | number[]
  | number[][]
  | number[][][]
  | number[][][][]
  | number[][][][][]
  | number[][][][][][];

export type TypedArray = Float32Array | Int32Array | Uint8Array;

export type NumberArray<T extends DataType = DataType> =
  | DataTypeArray<T>
  | Array<number>
  // deno-lint-ignore no-explicit-any
  | any;

/**
 * DataSet is a container for training data.
 */
export type DataSet = {
  inputs: Tensor2D;
  outputs: Tensor1D;
};

/** @docalias TypedArray|Array */
export type TensorLike =
  | TypedArray
  | number[][]
  | number[]
  | ArrayMap
  | TypedArray[];

export type ScalarLike = number | Uint8Array;

/** @docalias TypedArray|Array */
export type TensorLike1D =
  | TypedArray
  | number[]
  | Uint8Array[];

/** @docalias TypedArray|Array */
export type TensorLike2D =
  | TypedArray
  | number[]
  | number[][]
  | Uint8Array[]
  | Uint8Array[][];

/** @docalias TypedArray|Array */
export type TensorLike3D =
  | TypedArray
  | number[]
  | number[][][]
  | Uint8Array[]
  | Uint8Array[][][];
