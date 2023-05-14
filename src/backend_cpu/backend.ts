import { Rank, Shape } from "../core/api/shape.ts";
import { Tensor } from "../core/tensor/tensor.ts";
import { length } from "../core/tensor/util.ts";
import { Backend, DataSet, NetworkConfig } from "../core/types.ts";
import { Library } from "./mod.ts";
import {
  Buffer,
  encodeDatasets,
  encodeJSON,
  PredictOptions,
  TrainOptions,
} from "./util.ts";

/**
 * CPU Backend.
 */
export class CPUBackend implements Backend {
  library: Library;
  outputShape: Shape[Rank];
  #id: bigint;

  constructor(
    library: Library,
    outputShape: Shape[Rank],
    id: bigint,
  ) {
    this.library = library;
    this.outputShape = outputShape;
    this.#id = id;
  }

  static create(config: NetworkConfig, library: Library) {
    const buffer = encodeJSON(config);
    const shape = new Buffer();
    const id = library.symbols.ffi_backend_create(
      buffer,
      buffer.length,
      shape.allocBuffer,
    ) as bigint;
    const outputShape = Array.from(shape.buffer.slice(1)) as Shape[Rank];

    return new CPUBackend(library, outputShape, id);
  }

  train(datasets: DataSet[], epochs: number, batches: number, rate: number) {
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
      buffer.byteLength,
      options,
      options.byteLength,
    );
  }

  //deno-lint-ignore require-await
  async predict(input: Tensor<Rank>): Promise<Tensor<Rank>> {
    const options = encodeJSON({
      inputShape: [1, ...input.shape],
      outputShape: this.outputShape,
    } as PredictOptions);
    const output = new Float32Array(length(this.outputShape));
    this.library.symbols.ffi_backend_predict(
      this.#id,
      input.data as Float32Array,
      options,
      options.length,
      output,
    );
    return new Tensor(output, this.outputShape);
  }

  save(): Uint8Array {
    const shape = new Buffer();
    this.library.symbols.ffi_backend_save(this.#id, shape.allocBuffer);
    return shape.buffer;
  }

  saveFile(path: string): void {
    Deno.writeFileSync(path, this.save())
  }

  static load(buffer: Uint8Array, library: Library): CPUBackend {
    const shape = new Buffer();
    const id = library.symbols.ffi_backend_load(
      buffer,
      buffer.length,
      shape.allocBuffer,
    ) as bigint;
    const outputShape = Array.from(shape.buffer.slice(1)) as Shape[Rank];

    return new CPUBackend(library, outputShape, id);
  }

  static loadFile(path: string, library: Library): CPUBackend {
    return this.load(Deno.readFileSync(path), library)
  }
}
