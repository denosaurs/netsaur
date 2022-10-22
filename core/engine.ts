import { CPUBackend } from "../backends/cpu/backend.ts";
import { Backend, NetworkConfig } from "./types.ts";

export class Engine {
    static backendLoader: (config: NetworkConfig) => Backend = (config: NetworkConfig) => new CPUBackend(config);
}