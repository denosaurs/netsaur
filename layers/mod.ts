import { ConvCPULayer } from "../backends/cpu/layers/conv.ts";
import { DenseCPULayer } from "../backends/cpu/layers/dense.ts";
import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";
import {
  ConvLayerConfig,
  DenseLayerConfig,
  PoolLayerConfig,
} from "../core/types.ts";

export class Layer {
  // deno-lint-ignore no-explicit-any
  static layers: any = {
    dense: (config: DenseLayerConfig) => new DenseCPULayer(config),
    conv: (config: ConvLayerConfig) => new ConvCPULayer(config),
    pool: (config: PoolLayerConfig) => new PoolCPULayer(config),
  };
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
  if (Layer.layers.conv === undefined) {
    throw new Error("Current backend does not support pooling layers");
  }
  return Layer.layers.pool(config);
}
