import { CPUBackend } from "../backends/cpu/backend.ts";
import * as CPUKernels from "../backends/cpu/kernels/mod.ts";
import { Backend, NetworkConfig } from "./types.ts";

export class Engine {
  static backendLoader: (config: NetworkConfig) => Backend = (
    config: NetworkConfig,
  ) => new CPUBackend(config);
  // deno-lint-ignore no-explicit-any
  static kernels: any = CPUKernels;
}
