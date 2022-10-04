import ffi from "./ffi.ts";

const {
  layer_dense,
} = ffi;

enum C_ACTIVATION {
  sigmoid = 0,
  tanh = 1,
  relu = 2,
  linear = 3,
  leaky_relu = 4,
  relu6 = 5,
  elu = 6,
  selu = 7,
}

export type Activation = keyof typeof C_ACTIVATION;

export interface DenseLayerConfig {
  activation: Activation;
  units: number;
}

export class Layer {
  #ptr: Deno.PointerValue;

  get unsafePointer() {
    return this.#ptr;
  }

  constructor(ptr: Deno.PointerValue) {
    this.#ptr = ptr;
  }

  static dense(config: DenseLayerConfig): Layer {
    return new Layer(
      layer_dense(config.units, C_ACTIVATION[config.activation]),
    );
  }
}
