import { Engine } from "../../core/engine.ts";
import {
  BackendType,
  DenseLayerConfig,
  NetworkConfig,
} from "../../core/types.ts";
import { Layer } from "../../layers/mod.ts";
import { Tensor } from "../../mod.ts";

import { NativeBackend } from "./backend.ts";
import { DenseNativeLayer } from "./layers/dense.ts";

const loadBackend = (config: NetworkConfig) => new NativeBackend(config);

const dense = (config: DenseLayerConfig) => new DenseNativeLayer(config);
const layers = {
  dense,
};

const setup = (_silent = false) => {
  Tensor.type = BackendType.Native;
  Engine.backendLoader = loadBackend;
  Layer.layers = layers;
};

export const Native = {
  setup,
  loadBackend,
  layers,
};
export * from "./backend.ts";
export * from "./matrix.ts";
