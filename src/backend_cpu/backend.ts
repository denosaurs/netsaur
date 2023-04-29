import { Rank, Shape, Tensor } from "../../mod.ts";
import { length } from "../core/tensor/util.ts";
import { BackendType, DataSet, NetworkConfig } from "../core/types.ts";
import { NetworkJSON } from "../model/types.ts";
import { Library } from "./mod.ts";
import {
  encodeDatasets,
  encodeJSON,
  PredictOptions,
  TrainOptions,
} from "./util.ts";

export class CPUBackend {
  config: NetworkConfig;
  library: Library;
  outputShape?: Shape[Rank];

  constructor(config: NetworkConfig, library: Library) {
    this.config = config;
    this.library = library;

    const buffer = encodeJSON(config);
    this.library.symbols.ops_backend_create(buffer, buffer.length);
  }

  train(datasets: DataSet[], epochs: number, rate: number) {
    this.outputShape = datasets[0].outputs.shape.slice(1) as Shape[Rank];
    const buffer = encodeDatasets(datasets);
    const options = encodeJSON({
      datasets: datasets.length,
      inputShape: datasets[0].inputs.shape,
      outputShape: datasets[0].outputs.shape,
      epochs,
      rate,
    } as TrainOptions);

    this.library.symbols.ops_backend_train(
      buffer,
      buffer.byteLength,
      options,
      options.byteLength,
    );
  }

  //deno-lint-ignore require-await
  async predict(
    input: Tensor<Rank, BackendType>,
  ): Promise<Tensor<Rank, BackendType>> {
    const options = encodeJSON({
      inputShape: input.shape,
      outputShape: this.outputShape,
    } as PredictOptions);
    const output = new Float32Array(length(this.outputShape!));
    this.library.symbols.ops_backend_predict(
      input.data as Float32Array,
      options,
      options.length,
      output,
    );
    return new Tensor(output, this.outputShape!);
  }

  save(_input: string): void {}

  //deno-lint-ignore require-await
  async toJSON() {
    return null as unknown as NetworkJSON;
  }

  static fromJSON(_json: NetworkJSON): CPUBackend {
    return null as unknown as CPUBackend;
  }
}
