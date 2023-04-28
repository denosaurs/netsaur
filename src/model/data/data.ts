import { CsvLoaderConfig, loadCsv } from "./csv.ts";

export interface DataLike {
  // Model input data
  // deno-lint-ignore no-explicit-any
  train_x: any;
  // Model output data / labels
  // deno-lint-ignore no-explicit-any
  train_y: any;
  // Model test input data
  // deno-lint-ignore no-explicit-any
  test_x?: any;
  // Model test output data / labels
  // deno-lint-ignore no-explicit-any
  test_y?: any;
}

export class Data {
  // deno-lint-ignore no-explicit-any
  inputs: any;
  // deno-lint-ignore no-explicit-any
  outputs: any;

  constructor(data: DataLike) {
    this.inputs = data.train_x;
    this.outputs = data.train_y;
  }

  static async csv(url: string | URL, config?: CsvLoaderConfig) {
    return new Data(await loadCsv(url, config));
  }
}
