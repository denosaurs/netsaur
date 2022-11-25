import { DataType, DataTypeArray, WebGPUData } from "../deps.ts";
import { ConvCPULayer } from "../backends/cpu/layers/conv.ts";
import { DenseCPULayer } from "../backends/cpu/layers/dense.ts";
import { DenseGPULayer } from "../backends/gpu/layers/dense.ts";
import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";
import { Tensor } from "./tensor.ts";
import { Matrix } from "../backends/native/matrix.ts";
import { SoftmaxCPULayer } from "../backends/cpu/layers/activation.ts";

export interface LayerJSON {
  outputSize?: Shape[Rank];
  inputSize?: Shape[Rank];
  activationFn?: string;
  costFn?: string;
  type: string;
  weights?: TensorJSON;
  paddedSize?: Shape[Rank];
  biases?: TensorJSON;
  kernel?: TensorJSON;
  strides?: Shape[Rank];
  padding?: number;
  mode?: "max" | "avg";
}

export interface NetworkJSON {
  costFn?: string;
  input: Shape[Rank] | undefined;
  layers: LayerJSON[];
}

export interface TensorJSON {
  data: number[];
  shape: Shape[Rank];
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
  toJSON(): Promise<NetworkJSON>;
  getWeights(): Tensor<Rank, BackendType>[];
  getBiases(): Tensor<Rank, BackendType>[];
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

export type CPULayer =
  | ConvCPULayer
  | DenseCPULayer
  | PoolCPULayer
  | SoftmaxCPULayer;

export type GPULayer = DenseGPULayer;

export interface DenseLayerConfig {
  init?: Init;
  size: Shape1D;
  activation?: Activation;
}

export interface ConvLayerConfig {
  init?: Init;
  activation?: Activation;
  kernel?: DataTypeArray;
  kernelSize: Shape4D;
  padding?: number;
  unbiased?: boolean;
  strides?: Shape2D;
}

export interface PoolLayerConfig {
  strides?: Shape2D;
  mode?: "max" | "avg";
}

export interface FlattenLayerConfig {
  size: Shape[Rank];
}

export type Shape1D = [number];
export type Shape2D = [number, number];
export type Shape3D = [number, number, number];
export type Shape4D = [number, number, number, number];
export type Shape5D = [number, number, number, number, number];
export type Shape6D = [number, number, number, number, number, number];

export enum Rank {
  R1 = 1, // Scalar   (magnitude only)
  R2 = 2, // Vector   (magnitude and direction)
  R3 = 3, // Matrix   (table of numbers)
  R4 = 4, // 3-Tensor (cube of numbers)
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

export type CPUTensor<R extends Rank> = Tensor<R, BackendType.CPU>;
export type GPUTensor<R extends Rank> = Tensor<R, BackendType.GPU>;

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
  Native = "native",
}

/** @docalias TypedArray|Array */
export type TensorLike =
  | number
  | number[]
  | number[][]
  | number[][][]
  | number[][][][]
  | TypedArray
  | TypedArray[]
  | TypedArray[][]
  | TypedArray[][][];

export interface TensorData {
  cpu: DataTypeArray;
  gpu: WebGPUData;
  native: Matrix<"f32">;
}

export type Init = "uniform" | "xavier" | "xaviern" | "kaiming";

export interface InitFn {
  init<R extends Rank, B extends BackendType>(
    input: Shape[Rank],
    weights: Shape[R],
    outputs: Shape[Rank],
  ): Tensor<R, B>;
}
