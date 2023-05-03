import { CPU } from "../backend_cpu/mod.ts";
import { NetworkJSON } from "../model/types.ts";
import { Rank, Shape } from "./api/shape.ts";
import { Tensor } from "./tensor/tensor.ts";
import { Backend, BackendType, NetworkConfig } from "./types.ts";

export interface BackendInstance {
  init(): Promise<void>;
}

/**
 * The Tensor Backend is responsible for creating and managing Tensors.
 */
export interface TensorBackend {
  zeroes<R extends Rank, B extends BackendType>(shape: Shape[R]): Tensor<R, B>;

  from<R extends Rank, B extends BackendType>(
    values: Float32Array,
    shape: Shape[R],
  ): Tensor<R, B>;

  get(tensor: Tensor<Rank, BackendType>): Promise<Float32Array>;

  set(tensor: Tensor<Rank, BackendType>, values: Float32Array): void;
}

/**
 * The Backend Loader is responsible for loading the backend and setting it up.
 */
export interface BackendLoader {
  setup(silent: boolean): Promise<boolean>;

  loadBackend(config: NetworkConfig): Backend;

  fromJSON(json: NetworkJSON): Backend;
}

/**
 * setupBackend loads the backend and sets it up.
 */
export async function setupBackend(loader: BackendLoader, silent = false) {
  const success = await loader.setup(silent);
  if (!success) {
    if (!silent) console.log("Defaulting to CPU Backend");
    await CPU.setup(silent);
  }
  Engine.backendLoader = loader;
}

/**
 * the Engine manages the backend and the backend loader.
 */
export class Engine {
  static backendLoader: BackendLoader;

  static type: BackendType;
}
