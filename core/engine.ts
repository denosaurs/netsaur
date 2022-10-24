import { CPUBackend } from "../backends/cpu/backend.ts";
import * as CPUOps from "../backends/cpu/ops/mod.ts";
import { Backend, NetworkConfig } from "./types.ts";

export class Engine {
  static backendLoader: (config: NetworkConfig) => Backend = (
    config: NetworkConfig,
  ) => new CPUBackend(config);
  // deno-lint-ignore no-explicit-any
  static ops: any = CPUOps;
}
