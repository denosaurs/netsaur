import { Activation, Init, LayerType } from "../types.ts";
import { Rank, Shape, Shape1D, Shape2D, Shape4D } from "./shape.ts";

export type Layer =
  | { type: LayerType.Dense; config: DenseLayerConfig }
  | { type: LayerType.Activation; config: ActivationLayerConfig }
  | { type: LayerType.Conv; config: ConvLayerConfig }
  | { type: LayerType.Pool; config: PoolLayerConfig }
  | { type: LayerType.Flatten; config: FlattenLayerConfig }
  | { type: LayerType.Softmax };

export type DenseLayerConfig = {
  init?: Init;
  size: Shape1D;
  activation?: Activation;
};

export type ActivationLayerConfig = {
  activation: Activation;
};

export type ConvLayerConfig = {
  init?: Init;
  activation?: Activation;
  kernel?: Float32Array;
  kernelSize: Shape4D;
  padding?: number;
  unbiased?: boolean;
  strides?: Shape2D;
};

export type PoolLayerConfig = {
  strides?: Shape2D;
  mode?: "max" | "avg";
};

export type FlattenLayerConfig = {
  size: Shape[Rank];
};
