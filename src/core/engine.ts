import { CPU } from "../backend_cpu/mod.ts";
import { NetworkJSON } from "../model/types.ts";
import { Rank, Shape } from "./api/shape.ts";
import { Tensor } from "./tensor/tensor.ts";
import { Backend, BackendType, NetworkConfig } from "./types.ts";

export interface BackendInstance {
  init(): Promise<void>;
}

export interface TensorBackend {
  zeroes<R extends Rank, B extends BackendType>(shape: Shape[R]): Tensor<R, B>;

  from<R extends Rank, B extends BackendType>(
    values: Float32Array,
    shape: Shape[R],
  ): Tensor<R, B>;

  get(tensor: Tensor<Rank, BackendType>): Promise<Float32Array>;

  set(tensor: Tensor<Rank, BackendType>, values: Float32Array): void;
}

export interface BackendLoader {
  setup(silent: boolean): Promise<boolean>;

  loadBackend(config: NetworkConfig): Backend;

  fromJSON(json: NetworkJSON): Backend;
}

export async function setupBackend(loader: BackendLoader, silent = false) {
  const success = await loader.setup(silent);
  if (!success) {
    if (!silent) console.log("Defaulting to CPU Backend");
    await CPU.setup(silent);
  }
  Engine.backendLoader = loader
}

export class Engine {
  static backendLoader: BackendLoader;

  static type: BackendType;
}
