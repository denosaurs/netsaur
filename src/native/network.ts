import ffi, { cstr } from "./ffi.ts";
import { Layer } from "./layer.ts";
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

enum C_COST {
  crossentropy = 0,
  mse = 1,
}

export type Cost = keyof typeof C_COST;

export interface NetworkConfig {
  inputSize: number;
  layers: Layer[];
  cost: Cost;
}

export interface Dataset {
  inputs: Matrix<"f32">;
  outputs: Matrix<"f32">;
}

export class Network {
  #ptr: Deno.PointerValue;
  #token: { ptr: Deno.PointerValue } = { ptr: 0 };

  get unsafePointer() {
    return this.#ptr;
  }

  constructor(config: NetworkConfig | Deno.PointerValue) {
    this.#ptr = typeof config === "object"
      ? network_create(
        config.inputSize,
        C_COST[config.cost],
        config.layers.length,
        new BigUint64Array(config.layers.map((e) => BigInt(e.unsafePointer))),
      )
      : config;
    this.#token.ptr = this.#ptr;
    NetworkFinalizer.register(this, this.#ptr, this.#token);
  }

  predict(input: Matrix<"f32">): Matrix<"f32"> {
    return new Matrix(network_feed_forward(this.#ptr, input.unsafePointer));
  }

  train(datasets: Dataset[], epochs: number, learningRate: number) {
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
      learningRate,
    );
  }

  static load(path: string): Network {
    return new Network(network_load(cstr(path)));
  }

  save(path: string) {
    network_save(this.#ptr, cstr(path));
  }

  free(): void {
    if (this.#ptr) {
      network_free(this.#ptr);
      this.#ptr = 0;
      NetworkFinalizer.unregister(this.#token);
    }
  }
}
