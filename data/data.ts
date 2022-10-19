import { CsvLoaderConfig, loadCsv } from "./csv.ts";

export interface DataLike {
  // Model input data
  xs: any;
  // Model output data / labels
  ys: any;
}

export class Data {
  inputs: any;
  outputs: any;

  constructor(data: DataLike) {
    this.inputs = data.xs;
    this.outputs = data.ys;
  }

  static async csv(url: string | URL, config?: CsvLoaderConfig) {
    return new Data(await loadCsv(url, config));
  }
}
