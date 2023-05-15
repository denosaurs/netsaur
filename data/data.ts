// deno-lint-ignore-file no-explicit-any
import { Tensor } from "../mod.ts";
import { CsvLoaderConfig, loadCsv } from "./csv.ts";
import type { DataLike } from "./types.ts";

export class Data {
  /**
   * Model input data
   */
  inputs: Tensor<any>;

  /**
   * Model output data / labels
   */
  outputs: Tensor<any>;

  constructor(data: DataLike) {
    this.inputs = data.train_x;
    this.outputs = data.train_y;
  }

  /**
   * Load data from a CSV file or URL containing CSV data.
   */
  static async csv(url: string | URL, config?: CsvLoaderConfig) {
    return new Data(await loadCsv(url, config));
  }
}
