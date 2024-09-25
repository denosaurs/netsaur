import type { Rank, Shape } from "../../core/api/shape.ts";
import type { Backend, DataSet, NetworkConfig } from "../../core/types.ts";
import type { Library } from "./mod.ts";
import { Tensor } from "../../core/tensor/tensor.ts";
import { length } from "../../core/tensor/util.ts";
import {
  Buffer,
  encodeDatasets,
  encodeJSON,
  type PredictOptions,
  type TrainOptions,
} from "./util.ts";
import type { PostProcessor } from "../../core/api/postprocess.ts";

/**
 * CPU Backend.
 */
export class CPUBackend implements Backend {
  library: Library;
  outputShape: Shape<Rank>;
  #id: bigint;

  constructor(
    library: Library,
    outputShape: Shape<Rank>,
    id: bigint,
  ) {
    this.library = library;
    this.outputShape = outputShape;
    this.#id = id;
  }

  static create(config: NetworkConfig, library: Library): CPUBackend {
    const buffer = encodeJSON(config);
    const shape = new Buffer();
    const id = library.symbols.ffi_backend_create(
      buffer,
      BigInt(buffer.length),
      shape.allocBuffer,
    ) as bigint;
    const outputShape = Array.from(
      new Uint32Array(shape.buffer.slice(4).buffer),
    ) as Shape<Rank>;
    return new CPUBackend(library, outputShape, id);
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

  async predict(input: Tensor<Rank>, config: {postProcess: PostProcessor, outputShape?: Shape<Rank>}): Promise<Tensor<Rank>>;
  async predict(
    input: Tensor<Rank>,
    config: {postProcess: PostProcessor, outputShape?: Shape<Rank>},
    layers: number[],
  ): Promise<Tensor<Rank>>;
  //deno-lint-ignore require-await
  async predict(
    input: Tensor<Rank>,
    config: {postProcess: PostProcessor, outputShape?: Shape<Rank>},
    layers?: number[],    
  ): Promise<Tensor<Rank>> {
    const options = encodeJSON({
      inputShape: input.shape,
      outputShape: [input.shape[0], ...(config.outputShape ?? this.outputShape)],
      postProcess: config.postProcess,
      layers,
    } as PredictOptions);
    const output = new Float32Array(
      input.shape[0] * length(config.outputShape ?? this.outputShape),
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
        ...(config.outputShape ?? this.outputShape),
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

  static load(buffer: Uint8Array, library: Library): CPUBackend {
    const shape = new Buffer();
    const id = library.symbols.ffi_backend_load(
      buffer,
      BigInt(buffer.length),
      shape.allocBuffer,
    ) as bigint;
    const outputShape = Array.from(shape.buffer.slice(1)) as Shape<Rank>;

    return new CPUBackend(library, outputShape, id);
  }

  static loadFile(path: string, library: Library): CPUBackend {
    return this.load(Deno.readFileSync(path), library);
  }
}
