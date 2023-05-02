import { DataSet, Rank, Shape } from "../../mod.ts";

export type TrainOptions = {
  datasets: number;
  inputShape: Shape[Rank];
  outputShape: Shape[Rank];
  epochs: number;
  rate: number;
};

export type PredictOptions = {
  inputShape: Shape[Rank];
  outputShape: Shape[Rank];
};

export function encodeJSON(json: unknown) {
  return new TextEncoder().encode(JSON.stringify(json));
}

export function pointer(arr: BufferSource) {
  return BigInt(Deno.UnsafePointer.value(Deno.UnsafePointer.of(arr)));
}

export function encodeDatasets(datasets: DataSet[]) {
  const pointers: bigint[] = [];
  for (const dataset of datasets) {
    pointers.push(pointer(dataset.inputs.data as Float32Array));
    pointers.push(pointer(dataset.outputs.data as Float32Array));
  }
  return new BigUint64Array(pointers);
}
