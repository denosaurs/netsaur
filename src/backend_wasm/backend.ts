import { Rank, Shape } from "../core/api/shape.ts";
import { Tensor } from "../core/tensor/tensor.ts";
import { Backend, DataSet, NetworkConfig } from "../core/types.ts";
import {
  wasm_backend_create,
  wasm_backend_load,
  wasm_backend_predict,
  wasm_backend_save,
  wasm_backend_train,
} from "./lib/netsaur.generated.js";
import { PredictOptions, TrainOptions } from "./utils.ts";

/**
 * Web Assembly Backend.
 */
export class WASMBackend implements Backend {
  outputShape: Shape[Rank];
  #id: number;

  constructor(outputShape: Shape[Rank], id: number) {
    this.outputShape = outputShape;
    this.#id = id;
  }

  static create(config: NetworkConfig) {
    const shape = Array(0);
    const id = wasm_backend_create(JSON.stringify(config), shape);
    return new WASMBackend(shape as Shape[Rank], id);
  }

  train(datasets: DataSet[], epochs: number, batches: number, rate: number) {
    this.outputShape = datasets[0].outputs.shape.slice(1) as Shape[Rank];
    const buffer = [];
    for (const dataset of datasets) {
      buffer.push(dataset.inputs.data as Float32Array);
      buffer.push(dataset.outputs.data as Float32Array);
    }
    const options = JSON.stringify({
      datasets: datasets.length,
      inputShape: datasets[0].inputs.shape,
      outputShape: datasets[0].outputs.shape,
      epochs,
      batches,
      rate,
    } as TrainOptions);

    wasm_backend_train(this.#id, buffer, options);
  }

  //deno-lint-ignore require-await
  async predict(input: Tensor<Rank>): Promise<Tensor<Rank>> {
    const options = JSON.stringify({
      inputShape: [1, ...input.shape],
      outputShape: this.outputShape,
    } as PredictOptions);
    const output = wasm_backend_predict(
      this.#id,
      input.data as Float32Array,
      options,
    );
    return new Tensor(output, this.outputShape!);
  }

  save(): Uint8Array {
    return wasm_backend_save(this.#id);
  }

  saveFile(input: string): void {
    Deno.writeFileSync(input, this.save());
  }

  static loadFile(path: string): WASMBackend {
    return this.load(Deno.readFileSync(path));
  }

  static load(input: Uint8Array): WASMBackend {
    const shape = Array(0);
    const id = wasm_backend_load(input, shape);
    return new WASMBackend(shape as Shape[Rank], id);
  }
}
