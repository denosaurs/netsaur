import {
  ConvLayerConfig,
  DenseLayerConfig,
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
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.sigmoid();
}

export function LeakyReluLayer() {
  if (Layer.layers.softmax === undefined) {
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.softmax();
}

export function TanhLayer() {
  if (Layer.layers.softmax === undefined) {
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.tanh();
}

export function ReluLayer() {
  if (Layer.layers.softmax === undefined) {
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.relu();
}

export function Relu6Layer() {
  if (Layer.layers.softmax === undefined) {
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.softmax();
}

export function EluLayer() {
  if (Layer.layers.softmax === undefined) {
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.softmax();
}

export function SeluLayer() {
  if (Layer.layers.softmax === undefined) {
    throw new Error("Current backend does not support softmax layers");
  }
  return Layer.layers.softmax();
}


