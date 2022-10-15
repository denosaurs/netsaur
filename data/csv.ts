import { tensor2D } from "../core/tensor.ts";
import type { DataLike } from "./data.ts";
import { CsvStream } from "./deps.ts";

export interface CsvColumnConfig {
  label?: boolean;
}

export interface CsvLoaderConfig {
  columns?: Record<string, CsvColumnConfig>;
}

export async function loadCsv(
  url: string | URL,
  config: CsvLoaderConfig = {},
): Promise<DataLike> {
  const data = await fetch(url).then((res) =>
    res.body!.pipeThrough(new TextDecoderStream())
      .pipeThrough(new CsvStream())
  );
  const colConfigs = Object.entries(config.columns ?? {});
  const xs: number[][] = [];
  const ys: number[][] = [];
  const labelCols = colConfigs.filter(([, col]) => col.label);
  if (labelCols.length === 0) {
    throw new Error("No label column was set");
  }
  let columnNames!: string[];
  let columnIndices!: Record<string, number>;
  for await (const row of data) {
    if (!columnNames) {
      columnNames = row;
      columnIndices = columnNames.reduce((acc, col, i) => {
        acc[col] = i;
        return acc;
      }, {} as Record<string, number>);
      continue;
    }
    const x = [];
    const y = [];
    for (const col in columnIndices) {
      const colConfig = config.columns?.[col];
      const i = columnIndices[col];
      const value = row[i];
      if (colConfig?.label) {
        y.push(Number(value));
      } else {
        x.push(Number(value));
      }
    }
    xs.push(x);
    ys.push(y);
  }
  return {
    xs: await tensor2D(xs),
    ys: await tensor2D(ys),
  };
}
