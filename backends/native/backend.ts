import {
  Backend,
  DenseLayerConfig,
  Layer,
  NetworkConfig,
  Size,
} from "../../core/types.ts";
import { to1D } from "../../core/util.ts";
import ffi, { cstr } from "./ffi.ts";
import { DenseNativeLayer } from "./layers/dense.ts";
import { Matrix } from "./matrix.ts";

const {
  network_free,
  network_create,
  network_feed_forward,
  network_train,
  network_load,
  network_save,
} = ffi;

const NetworkFinalizer = new FinalizationRegistry(
  (network: Deno.PointerValue) => {
    network_free(network);
  },
);

// const C_COST: { [key: string]: number } = {
//   crossentropy: 1,
//   mse: 0,
// };

export type NativeLayer = DenseNativeLayer;

export interface Dataset {
  inputs: Matrix<"f32">;
  outputs: Matrix<"f32">;
}

export class NativeBackend implements Backend {
  #ptr: Deno.PointerValue;
  #token: { ptr: Deno.PointerValue } = { ptr: 0 };

  get unsafePointer() {
    return this.#ptr;
  }
  get layers() {
    // TODO: get layers from backend
    return [];
  }
  constructor(config: NetworkConfig | Deno.PointerValue) {
    this.#ptr = typeof config === "object"
      ? network_create(
        to1D(config.input!),
        config.cost === "crossentropy" ? 1 : 0,
        config.layers.length,
        new BigUint64Array(
          config.layers.map((e) => BigInt(e.unsafePointer)),
        ),
      )
      : config;
    this.#token.ptr = this.#ptr;
    NetworkFinalizer.register(this, this.#ptr, this.#token);
  }

  addLayer(_layer: Layer): void {
  }

  encodeLayer(layer: Layer): NativeLayer {
    switch (layer.type) {
      case "dense":
        return (new DenseNativeLayer(layer.config as DenseLayerConfig));
      default:
        throw new Error(
          `${
            layer.type.charAt(0).toUpperCase() + layer.type.slice(1)
          }Layer not implemented for the CPU backend`,
        );
    }
  }

  initialize(_inputSize: Size, _batches: number): void {
  }

  feedForward(input: Matrix<"f32">): Matrix<"f32"> {
    return new Matrix(network_feed_forward(this.#ptr, input.unsafePointer));
  }

  train(datasets: Dataset[], epochs = 5000, _batches = 1, rate = 0.1) {
    const datasetBuffers = datasets.map((e) =>
      new BigUint64Array(
        [e.inputs.unsafePointer, e.outputs.unsafePointer].map(BigInt),
      )
    );
    const datasetBufferPointers = new BigUint64Array(
      datasetBuffers.map((e) => BigInt(Deno.UnsafePointer.of(e))),
    );
    network_train(
      this.unsafePointer,
      datasets.length,
      datasetBufferPointers,
      epochs,
      rate,
    );
  }

  predict(input: Matrix<"f32">): Matrix<"f32"> {
    return new Matrix(network_feed_forward(this.#ptr, input.unsafePointer));
  }

  save(path: string) {
    network_save(this.#ptr, cstr(path));
  }

  static load(path: string): NativeBackend {
    return new NativeBackend(network_load(cstr(path)));
  }

  free(): void {
    if (this.#ptr) {
      network_free(this.#ptr);
      this.#ptr = 0;
      NetworkFinalizer.unregister(this.#token);
    }
  }

  toJSON(): undefined {
    return;
  }

  getWeights() {
    return [];
  }

  getBiases() {
    return [];
  }
}
