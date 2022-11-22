import { Engine } from "../../core/engine.ts";
import {
  Backend,
  BackendType,
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
import * as kernels from "./kernels/mod.ts";
import {
  ReluCPULayer,
  SigmoidCPULayer,
  SoftmaxCPULayer,
  TanhCPULayer,
} from "./layers/activation.ts";

const loadBackend = (config: NetworkConfig): Backend => {
  return new CPUBackend(config);
};

// deno-lint-ignore require-await
const model = async (data: NetworkJSON, _silent = false): Promise<Backend> =>
  CPUBackend.fromJSON(data);

const dense = (config: DenseLayerConfig) => new DenseCPULayer(config);
const conv = (config: ConvLayerConfig) => new ConvCPULayer(config);
const pool = (config: PoolLayerConfig) => new PoolCPULayer(config);
const softmax = () => new SoftmaxCPULayer();
const sigmoid = () => new SigmoidCPULayer();
const tanh = () => new TanhCPULayer();
const relu = () => new ReluCPULayer();

const layers = {
  dense,
  conv,
  pool,
  softmax,
  sigmoid,
  tanh,
  relu,
};

const setup = (_silent = false) => {
  Tensor.type = BackendType.CPU;
  Engine.backendLoader = loadBackend;
  Layer.layers = layers;
  Engine.kernels = kernels;
};

export const CPU = {
  setup,
  loadBackend,
  model,
  layers,
  kernels,
};
export { CPUBackend };
export * from "./kernels/matrix.ts";
export type { DataSet } from "../../core/types.ts";
