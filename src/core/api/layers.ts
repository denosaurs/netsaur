import { Activation, LayerType } from "../types.ts";
import {
  ConvLayerConfig,
  DenseLayerConfig,
  FlattenLayerConfig,
  Layer,
  PoolLayerConfig,
} from "./layer.ts";

/**
 * Creates a dense layer.
 */
export function DenseLayer(config: DenseLayerConfig): Layer {
  return { type: LayerType.Dense, config };
}

/**
 * Creates a convolutional layer.
 */
export function ConvLayer(config: ConvLayerConfig): Layer {
  return { type: LayerType.Conv, config };
}

/**
 * Creates a pooling layer.
 */
export function PoolLayer(config: PoolLayerConfig): Layer {
  return { type: LayerType.Pool, config };
}

/**
 * Creates a softmax layer.
 */
export function SoftmaxLayer(): Layer {
  return { type: LayerType.Softmax };
}

/**
 * Creates a sigmoid layer.
 */
export function SigmoidLayer(): Layer {
  const config = { activation: Activation.Sigmoid };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a leaky relu layer.
 */
export function LeakyReluLayer(): Layer {
  const config = { activation: Activation.LeakyRelu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a tanh layer.
 */
export function TanhLayer(): Layer {
  const config = { activation: Activation.Tanh };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a relu layer.
 */
export function ReluLayer(): Layer {
  const config = { activation: Activation.Relu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a relu6 layer.
 */
export function Relu6Layer(): Layer {
  const config = { activation: Activation.Relu6 };
  return { type: LayerType.Activation, config };
}

/**
 * Creates an Elu layer.
 */
export function EluLayer(): Layer {
  const config = { activation: Activation.Elu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a Selu layer.
 */
export function SeluLayer(): Layer {
  const config = { activation: Activation.Selu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a Flatten layer.
 */
export function FlattenLayer(config: FlattenLayerConfig): Layer {
  return { type: LayerType.Flatten, config };
}
