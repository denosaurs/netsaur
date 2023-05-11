import { Rank, Shape, Tensor } from "../../mod.ts";
import { length } from "../core/tensor/util.ts";
import { Backend, DataSet, NetworkConfig } from "../core/types.ts";
import { Library } from "./mod.ts";
import {
  encodeDatasets,
  encodeJSON,
  PredictOptions,
  TrainOptions,
} from "./util.ts";

/**
 * CPU Backend.
 */
export class CPUBackend implements Backend {
  config: NetworkConfig;
  library: Library;
  outputShape: Shape[Rank];

  constructor(config: NetworkConfig, library: Library) {
    this.config = config;
    this.library = library;

    const buffer = encodeJSON(config);
    const shape = new Uint8Array(6);
    const length = this.library.symbols.ffi_backend_create(
      buffer,
      buffer.length,
      shape,
    );
    this.outputShape = Array.from(shape.slice(1, length)) as Shape[Rank];
  }

  train(datasets: DataSet[], epochs: number, rate: number) {
    const buffer = encodeDatasets(datasets);
    const options = encodeJSON({
      datasets: datasets.length,
      inputShape: datasets[0].inputs.shape,
      outputShape: datasets[0].outputs.shape,
      epochs,
      rate,
    } as TrainOptions);

    this.library.symbols.ffi_backend_train(
      buffer,
      buffer.byteLength,
      options,
      options.byteLength,
    );
  }

  //deno-lint-ignore require-await
  async predict(input: Tensor<Rank>): Promise<Tensor<Rank>> {
    const options = encodeJSON({
      inputShape: input.shape,
      outputShape: this.outputShape,
    } as PredictOptions);
    const output = new Float32Array(length(this.outputShape));
    this.library.symbols.ffi_backend_predict(
      input.data as Float32Array,
      options,
      options.length,
      output,
    );
    return new Tensor(output, this.outputShape);
  }

  save(input: string): void {
    const ptr = new Deno.UnsafePointerView(
      this.library.symbols.ffi_backend_save()!,
    );
    const lengthBe = new Uint8Array(4);
    const view = new DataView(lengthBe.buffer);
    ptr.copyInto(lengthBe, 0);
    const buf = new Uint8Array(view.getUint32(0));
    ptr.copyInto(buf, 4);
    Deno.writeFileSync(input, buf);
  }

  static loadModel(_input: string | Uint8Array): CPUBackend {
    return null as unknown as CPUBackend;
  }
}
