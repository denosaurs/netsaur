import { Tensor } from "../tensor/tensor.ts";
import { Activation, Init, LayerType } from "../types.ts";
import { Rank, Shape, Shape1D, Shape2D, Shape4D } from "./shape.ts";

/**
 * Layer is the base type for all layers.
 */
export type Layer =
  | { type: LayerType.Activation; config: ActivationLayerConfig }
  | { type: LayerType.Conv2D; config: Conv2DLayerConfig }
  | { type: LayerType.Dense; config: DenseLayerConfig }
  | { type: LayerType.Dropout; config: DropoutLayerConfig }
  | { type: LayerType.Flatten; config: FlattenLayerConfig }
  | { type: LayerType.Pool2D; config: Pool2DLayerConfig }
  | { type: LayerType.Softmax };

/**
 * The configuration for a dense layer.
 */
export type DenseLayerConfig = {
  /**
   * The type of initialization to use.
   */
  init?: Init;

  /**
   * The size of the layer.
   */
  size: Shape1D;
};

/**
 * The configuration for a dropout layer.
 */
export type DropoutLayerConfig = {
  /**
   * probability of dropping out a value.
   */
  probability: number;

  /**
   * whether or not to do the operation in place.
   */
  inplace?: boolean;
};

/**
 * The configuration for an activation layer.
 */
export type ActivationLayerConfig = {
  /**
   * The activation function to use.
   */
  activation: Activation;
};

/**
 * The configuration for a convolutional layer.
 */
export type Conv2DLayerConfig = {
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
};

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
export type Pool2DLayerConfig = {
  /**
   * The optional strides to use.
   */
  strides?: Shape2D;

  /**
   * The mode to use for the pool layer.
   */
  mode?: PoolMode;
};

/**
 * The configuration for a flatten layer.
 */
export type FlattenLayerConfig = {
  /**
   * The size of the layer.
   */
  size: Shape[Rank];
};
