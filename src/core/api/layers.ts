import { Activation, LayerType } from "../types.ts";
import {
  ConvLayerConfig,
  DenseLayerConfig,
  FlattenLayerConfig,
  Layer,
  PoolLayerConfig,
} from "./layer.ts";

export function DenseLayer(config: DenseLayerConfig): Layer {
  return { type: LayerType.Dense, config };
}

export function ConvLayer(config: ConvLayerConfig): Layer {
  return { type: LayerType.Conv, config };
}

export function PoolLayer(config: PoolLayerConfig): Layer {
  return { type: LayerType.Pool, config };
}

export function SoftmaxLayer(): Layer {
  return { type: LayerType.Softmax };
}

export function SigmoidLayer(): Layer {
  const config = { activation: Activation.Sigmoid };
  return { type: LayerType.Activation, config };
}

export function LeakyReluLayer(): Layer {
  const config = { activation: Activation.LeakyRelu };
  return { type: LayerType.Activation, config };
}

export function TanhLayer(): Layer {
  const config = { activation: Activation.Tanh };
  return { type: LayerType.Activation, config };
}

export function ReluLayer(): Layer {
  const config = { activation: Activation.Relu };
  return { type: LayerType.Activation, config };
}

export function Relu6Layer(): Layer {
  const config = { activation: Activation.Relu6 };
  return { type: LayerType.Activation, config };
}

export function EluLayer(): Layer {
  const config = { activation: Activation.Elu };
  return { type: LayerType.Activation, config };
}

export function SeluLayer(): Layer {
  const config = { activation: Activation.Selu };
  return { type: LayerType.Activation, config };
}

export function FlattenLayer(config: FlattenLayerConfig): Layer {
  return { type: LayerType.Flatten, config };
}
