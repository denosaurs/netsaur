import { Engine } from "../../core/engine.ts";
import {
  Backend,
  ConvLayerConfig,
  DenseLayerConfig,
  NetworkConfig,
  NetworkJSON,
  PoolLayerConfig,
} from "../../core/types.ts";
import { Layer } from "../../layers/mod.ts";
import { Tensor } from "../../mod.ts";

import { CPUBackend } from "./backend.ts";
import { ConvCPULayer } from "./layers/conv.ts";
import { DenseCPULayer } from "./layers/dense.ts";
import { PoolCPULayer } from "./layers/pool.ts";
import { TensorCPUBackend } from "./tensor.ts";
import * as ops from "./ops/mod.ts"

const loadBackend = (config: NetworkConfig): Backend => {
  return new CPUBackend(config);
};

// deno-lint-ignore require-await
const model = async (data: NetworkJSON, _silent = false): Promise<Backend> =>
  CPUBackend.fromJSON(data);

const dense = (config: DenseLayerConfig) => new DenseCPULayer(config);
const conv = (config: ConvLayerConfig) => new ConvCPULayer(config);
const pool = (config: PoolLayerConfig) => new PoolCPULayer(config);

const layers = {
  dense,
  conv,
  pool,
};

const setup = (_silent = false) => {
  Tensor.backend = new TensorCPUBackend();
  Engine.backendLoader = loadBackend;
  Layer.layers = layers;
  Engine.ops = ops;
};

export const CPU = {
  setup,
  loadBackend,
  model,
  layers,
  ops
};
export { CPUBackend };
export * from "./matrix.ts";
export type { DataSet } from "../../core/types.ts";
