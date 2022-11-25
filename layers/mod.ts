import {
  ConvLayerConfig,
  DenseLayerConfig,
  FlattenLayerConfig,
  PoolLayerConfig,
} from "../core/types.ts";

export class Layer {
  // deno-lint-ignore no-explicit-any
  static layers: any = {};
}

export function DenseLayer(config: DenseLayerConfig) {
  if (Layer.layers.dense === undefined) {
    throw new Error("Current backend does not support dense layers");
  }
  return Layer.layers.dense(config);
}

export function ConvLayer(config: ConvLayerConfig) {
  if (Layer.layers.conv === undefined) {
    throw new Error("Current backend does not support convolutional layers");
  }
  return Layer.layers.conv(config);
}
export function PoolLayer(config: PoolLayerConfig) {
  if (Layer.layers.pool === undefined) {
    throw new Error("Current backend does not support pooling layers");
  }
  return Layer.layers.pool(config);
}

export function SoftmaxLayer() {
  if (Layer.layers.softmax === undefined) {
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.softmax();
}

export function SigmoidLayer() {
  if (Layer.layers.sigmoid === undefined) {
    throw new Error("Current backend does not support sigmoid layers");
  }
  return Layer.layers.sigmoid();
}

export function LeakyReluLayer() {
  if (Layer.layers.leakyrelu === undefined) {
    throw new Error("Current backend does not support leakyrelu layers");
  }
  return Layer.layers.leakyrelu();
}

export function TanhLayer() {
  if (Layer.layers.tanh === undefined) {
    throw new Error("Current backend does not support tanh layers");
  }
  return Layer.layers.tanh();
}

export function ReluLayer() {
  if (Layer.layers.relu === undefined) {
    throw new Error("Current backend does not support relu layers");
  }
  return Layer.layers.relu();
}

export function Relu6Layer() {
  if (Layer.layers.relu6 === undefined) {
    throw new Error("Current backend does not support relu6 layers");
  }
  return Layer.layers.relu6();
}

export function EluLayer() {
  if (Layer.layers.elu === undefined) {
    throw new Error("Current backend does not support elu layers");
  }
  return Layer.layers.elu();
}

export function SeluLayer() {
  if (Layer.layers.selu === undefined) {
    throw new Error("Current backend does not support selu layers");
  }
  return Layer.layers.selu();
}

export function FlattenLayer(config: FlattenLayerConfig) {
  if (Layer.layers.flatten === undefined) {
    throw new Error("Current backend does not support flatten layers");
  }
  return Layer.layers.flatten(config);
}