import type { Backend, DataSet, NetworkConfig } from "../../core/types.ts";
import type { Library } from "./mod.ts";
import { Tensor, type Shape } from "../../../../tensor/mod.ts";
import { length, } from "../../../../tensor/src/utils.ts";
import {
  Buffer,
  encodeDatasets,
  encodeJSON,
  type PredictOptions,
  type TrainOptions,
} from "./util.ts";
import type { Rank } from "../../../../tensor/src/types.ts";

/**
 * GPU Backend.
 */
export class GPUBackend implements Backend {
  library: Library;
  outputShape: Shape<Rank>;
  #id: bigint;

  constructor(library: Library, outputShape: Shape<Rank>, id: bigint) {
    this.library = library;
    this.outputShape = outputShape;
    this.#id = id;
  }

  static create(config: NetworkConfig, library: Library): GPUBackend {
    const buffer = encodeJSON(config);
    const shape = new Buffer();
    const id = library.symbols.ffi_backend_create(
      buffer,
      BigInt(buffer.length),
      shape.allocBuffer,
    ) as bigint;
    const outputShape = Array.from(shape.buffer.slice(1)) as Shape<Rank>;

    return new GPUBackend(library, outputShape, id);
  }

  train(
    datasets: DataSet[],
    epochs: number,
    batches: number,
    rate: number,
  ): void {
    const buffer = encodeDatasets(datasets);
    const options = encodeJSON({
      datasets: datasets.length,
      inputShape: datasets[0].inputs.shape,
      outputShape: datasets[0].outputs.shape,
      epochs,
      batches,
      rate,
    } as TrainOptions);

    this.library.symbols.ffi_backend_train(
      this.#id,
      buffer,
      BigInt(buffer.byteLength),
      options,
      BigInt(options.byteLength),
    );
  }

  async predict(input: Tensor<Rank>): Promise<Tensor<Rank>>;
  async predict(
    input: Tensor<Rank>,
    layers: number[],
    outputShape: Shape<Rank>,
  ): Promise<Tensor<Rank>>;
  //deno-lint-ignore require-await
  async predict(
    input: Tensor<Rank>,
    layers?: number[],
    outputShape?: Shape<Rank>,
  ): Promise<Tensor<Rank>> {
    const options = encodeJSON({
      inputShape: input.shape,
      outputShape: [input.shape[0], ...(outputShape ?? this.outputShape)],
      layers,
    } as PredictOptions);
    const output = new Float32Array(
      input.shape[0] * length(outputShape ?? this.outputShape),
    );
    this.library.symbols.ffi_backend_predict(
      this.#id,
      input.data as Float32Array,
      options,
      BigInt(options.length),
      output,
    );
    return new Tensor(
      output,
      [
        input.shape[0],
        ...(outputShape ?? this.outputShape),
      ] as Shape<Rank>,
    );
  }

  save(): Uint8Array {
    const shape = new Buffer();
    this.library.symbols.ffi_backend_save(this.#id, shape.allocBuffer);
    return shape.buffer;
  }

  saveFile(path: string): void {
    Deno.writeFileSync(path, this.save());
  }

  static load(buffer: Uint8Array, library: Library): GPUBackend {
    const shape = new Buffer();
    const id = library.symbols.ffi_backend_load(
      buffer,
      BigInt(buffer.length),
      shape.allocBuffer,
    ) as bigint;
    const outputShape = Array.from(shape.buffer.slice(1)) as Shape<Rank>;

    return new GPUBackend(library, outputShape, id);
  }

  static loadFile(path: string, library: Library): GPUBackend {
    return this.load(Deno.readFileSync(path), library);
  }
}
