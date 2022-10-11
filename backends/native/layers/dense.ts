import ffi from "../ffi.ts";

import { DenseLayerConfig } from "../../../core/types.ts";
import { to1D } from "../../../core/util.ts";

const {
  layer_dense,
} = ffi;

const C_ACTIVATION: { [key: string]: number } = {
  "sigmoid": 0,
  "tanh": 1,
  "relu": 2,
  "relu6": 5,
  "leakyrelu": 4,
  "elu": 6,
  "linear": 3,
  "selu": 7,
};
// export type Activation = keyof typeof C_ACTIVATION;

export class DenseNativeLayer {
  #ptr: Deno.PointerValue;

  get unsafePointer() {
    return this.#ptr;
  }

  constructor(config: DenseLayerConfig) {
    this.#ptr = layer_dense(to1D(config.size), C_ACTIVATION[config.activation]);
  }
}
