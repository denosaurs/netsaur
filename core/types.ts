import { DataType, DataTypeArray, WebGPUData } from "../deps.ts";
import { ConvCPULayer } from "../backends/cpu/layers/conv.ts";
import { DenseCPULayer } from "../backends/cpu/layers/dense.ts";
import { DenseGPULayer } from "../backends/gpu/layers/dense.ts";
import { GPUMatrix } from "../backends/gpu/matrix.ts";
import { CPUMatrix } from "../backends/cpu/kernels/matrix.ts";
import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";
import { Tensor } from "./tensor.ts";

export interface LayerJSON {
  outputSize: number | Shape[Rank.R2];
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
  strides?: Shape[Rank];
  padding?: number;
  mode?: "max" | "avg";
}

export interface NetworkJSON {
  costFn?: string;
  type: "NeuralNetwork";
  sizes: (number | Shape[Rank.R2])[];
  input: Shape[Rank] | undefined;
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
    inputSize: Shape[Rank],
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

/**
 * NetworkConfig represents the configuration of a neural network.
 */
export interface NetworkConfig {
  input?: Shape[Rank];
  layers: Layer[];
  cost: Cost;
  silent?: boolean;
}

// deno-lint-ignore no-explicit-any
export type Layer = any;

export type CPULayer = ConvCPULayer | DenseCPULayer | PoolCPULayer;

export type GPULayer = DenseGPULayer;

export interface DenseLayerConfig {
  size: Shape1D;
  activation?: Activation;
}

export interface ConvLayerConfig {
  activation?: Activation;
  kernel?: DataTypeArray;
  kernelSize: Shape2D;
  padding?: number;
  unbiased?: boolean;
  strides?: Shape2D;
}

export interface PoolLayerConfig {
  strides?: Shape2D;
  mode?: "max" | "avg";
}

export type Shape1D = [number];
export type Shape2D = [number, number];
export type Shape3D = [number, number, number];
export type Shape4D = [number, number, number, number];
export type Shape5D = [number, number, number, number, number];
export type Shape6D = [number, number, number, number, number, number];

export enum Rank {
  R1 = 1, // Scalar   (magnitude only)
  R2 = 2,	// Vector   (magnitude and direction)
  R3 = 3, // Matrix   (table of numbers)
  R4 = 4,	// 3-Tensor (cube of numbers)
  R5 = 5,
  R6 = 6,
}

export interface Shape {
  1: Shape1D;
  2: Shape2D;
  3: Shape3D;
  4: Shape4D;
  5: Shape5D;
  6: Shape6D;
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

export type CPUTensor<R extends Rank> = Tensor<R, BackendType.CPU>
export type GPUTensor<R extends Rank> = Tensor<R, BackendType.GPU>


/**
 * DataSet is a container for training data.
 */
export type DataSet = {
  inputs: Tensor<Rank, BackendType>;
  outputs: Tensor<Rank, BackendType>;
};

export enum BackendType {
  CPU = "cpu",
  GPU = "gpu",
  // Native = "native",
}

/** @docalias TypedArray|Array */
export type TensorLike =
  | number
  | number[]
  | number[][]
  | number[][][]
  | TypedArray
  | TypedArray[]
  | TypedArray[][]
  | TypedArray[][][]

export interface TensorData {
  cpu: DataTypeArray,
  gpu: WebGPUData
}
