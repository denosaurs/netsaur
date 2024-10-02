import type { Tensor } from "../tensor/tensor.ts";
import type { Activation, Init, LayerType } from "../types.ts";
import type { Rank, Shape, Shape1D, Shape2D, Shape3D, Shape4D } from "./shape.ts";

/**
 * Layer is the base type for all layers.
 */
export type Layer =
  | { type: LayerType.Activation; config: ActivationLayerConfig }
  | { type: LayerType.Conv1D; config: Conv1DLayerConfig }
  | { type: LayerType.Conv2D; config: Conv2DLayerConfig }
  | { type: LayerType.ConvTranspose1D; config: ConvTranspose1DLayerConfig }
  | { type: LayerType.ConvTranspose2D; config: ConvTranspose2DLayerConfig }
  | { type: LayerType.Dense; config: DenseLayerConfig }
  | { type: LayerType.Dropout1D; config: DropoutLayerConfig }
  | { type: LayerType.Dropout2D; config: DropoutLayerConfig }
  | { type: LayerType.Embedding; config: EmbeddingLayerConfig }
  | { type: LayerType.Flatten }
  | { type: LayerType.LSTM; config: LSTMLayerConfig }
  | { type: LayerType.Pool2D; config: Pool2DLayerConfig }
  | { type: LayerType.BatchNorm1D; config: BatchNormLayerConfig }
  | { type: LayerType.BatchNorm2D; config: BatchNormLayerConfig }
  | { type: LayerType.Softmax, config: SoftmaxLayerConfig };

/**
 * The configuration for an LSTM layer.
 */
export interface LSTMLayerConfig {
  /**
   * The type of initialization to use.
   */
  init?: Init;

  /**
   * Number of units in the layer.
   */
  size: number;

  /**
   * Inverse of regularization strength.
   */
  c?: number;

  /**
   * Ratio of l1:l2.
   */
  l1Ratio?: number;

  /** 
   * Whether to return all time steps.
   */
  returnSequences?: boolean;

  activation?: Activation;

  recurrentActivation?: Activation;
}

  /**
 * The configuration for a dense layer.
 */
export interface DenseLayerConfig {
  /**
   * The type of initialization to use.
   */
  init?: Init;

  /**
   * The size of the layer.
   */
  size: number;

  /**
   * Inverse of regularization strength.
   */
  c?: number;

  /**
   * Ratio of l1:l2.
   */
  l1Ratio?: number;
}

/**
 * The configuration for a dropout layer.
 */
export interface DropoutLayerConfig {
  /**
   * probability of dropping out a value.
   */
  probability: number;

  /**
   * whether or not to do the operation in place.
   */
  inplace?: boolean;
}

/**
 * The configuration for an activation layer.
 */
export interface ActivationLayerConfig {
  /**
   * The activation function to use.
   */
  activation: Activation;
}

/**
 * The configuration for a convolutional layer.
 */
export interface Conv1DLayerConfig {
  /**
   * The type of initialization to use.
   */
  init?: Init;

  /**
   * The kernel to use.
   */
  kernel?: Tensor<Rank>;

  /**
   * The size of the kernel.
   */
  kernelSize: Shape3D;

  /**
   * The optional padding to use.
   */
  padding?: Shape1D;

  /**
   * The optional strides to use.
   */
  strides?: Shape1D;

  /**
   * Inverse of regularization strength.
   */
  c?: number;

  /**
   * Ratio of l1:l2.
   */
  l1Ratio?: number;
}

/**
 * The configuration for a convolutional layer.
 */
export interface Conv2DLayerConfig {
  /**
   * The type of initialization to use.
   */
  init?: Init;

  /**
   * The kernel to use.
   */
  kernel?: Tensor<Rank>;

  /**
   * The size of the kernel.
   */
  kernelSize: Shape4D;

  /**
   * The optional padding to use.
   */
  padding?: Shape2D;

  /**
   * The optional strides to use.
   */
  strides?: Shape2D;

  /**
   * Inverse of regularization strength.
   */
  c?: number;

  /**
   * Ratio of l1:l2.
   */
  l1Ratio?: number;
}

/**
 * The configuration for a convolution transpose layer.
 */
export interface ConvTranspose1DLayerConfig {
  /**
   * The type of initialization to use.
   */
  init?: Init;

  /**
   * The kernel to use.
   */
  kernel?: Tensor<Rank>;

  /**
   * The size of the kernel.
   */
  kernelSize: Shape3D;

  /**
   * The optional padding to use.
   */
  padding?: Shape1D;

  /**
   * The optional strides to use.
   */
  strides?: Shape1D;

  /**
   * Inverse of regularization strength.
   */
  c?: number;

  /**
   * Ratio of l1:l2.
   */
  l1Ratio?: number;
}

/**
 * The configuration for a convolutional transpose layer.
 */
export interface ConvTranspose2DLayerConfig {
  /**
   * The type of initialization to use.
   */
  init?: Init;

  /**
   * The size of the kernel.
   */
  kernelSize: Shape4D;

  /**
   * The optional padding to use.
   */
  padding?: Shape2D;

  /**
   * The optional strides to use.
   */
  strides?: Shape2D;

  /**
   * Inverse of regularization strength.
   */
  c?: number;

  /**
   * Ratio of l1:l2.
   */
  l1Ratio?: number;
}

export enum PoolMode {
  /**
   * The average pooling mode.
   */
  Avg,

  /**
   * The max pooling mode.
   */
  Max,
}

/**
 * The configuration for a pooling layer.
 */
export interface Pool2DLayerConfig {
  /**
   * The optional strides to use.
   */
  strides?: Shape2D;

  /**
   * The mode to use for the pool layer.
   */
  mode?: PoolMode;
}

/**
 * The configuration for an embedding layer.
 */
export interface EmbeddingLayerConfig {

  /**
   * Size of each embedding vector.
   */
  embeddingSize: number;

  /**
   * Number of words in the vocabulary.
   */
  vocabSize: number;

  /**
   * Inverse of regularization strength.
   */
  c?: number;

  /**
   * Ratio of l1:l2.
   */
  l1Ratio?: number;
}

/**
 * The configuration for a flatten layer.
 */
export interface FlattenLayerConfig {
  /**
   * The size of the layer.
   */
  size: Shape<Rank>;
}

/**
 * The configuration for a batch normalization layer.
 */
export interface BatchNormLayerConfig {
  /**
   * The momentum to use for the batch normalization.
   * Defaults to 0.99.
   * https://arxiv.org/abs/1502.03167
   */
  momentum?: number;

  /**
   * The epsilon to use for the batch normalization.
   * Defaults to 0.001.
   * https://arxiv.org/abs/1502.03167
   */
  epsilon?: number;
}


/**
 * The configuration for a softmax layer.
 */
export interface SoftmaxLayerConfig {
  /**
   * A temperature for scaling softmax inputs
   * to prevent exponential overflow/underflow
   */
  temperature?: number;
}
