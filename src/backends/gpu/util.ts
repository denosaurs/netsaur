import { Rank, Shape } from "../../core/api/shape.ts";
import { DataSet } from "../../core/types.ts";

export class Buffer {
  buffer = new Uint8Array();
  allocBuffer = new Deno.UnsafeCallback({
    parameters: ["usize"],
    result: "buffer",
  }, (length) => {
    this.buffer = new Uint8Array(Number(length));
    return this.buffer;
  }).pointer;
}

/**
 * Train Options Interface.
 */
export type TrainOptions = {
  datasets: number;
  inputShape: Shape[Rank];
  outputShape: Shape[Rank];
  epochs: number;
  batches: number;
  rate: number;
};

/**
 * Predict Options Interface.
 */
export type PredictOptions = {
  inputShape: Shape[Rank];
  outputShape: Shape[Rank];
};

/**
 * Encode JSON data.
 */
export function encodeJSON(json: unknown) {
  return new TextEncoder().encode(JSON.stringify(json));
}

/**
 * Returns the BigInt value of a pointer.
 */
export function pointer(arr: BufferSource) {
  return BigInt(Deno.UnsafePointer.value(Deno.UnsafePointer.of(arr)));
}

/**
 * Encode datasets.
 */
export function encodeDatasets(datasets: DataSet[]) {
  const pointers: bigint[] = [];
  for (const dataset of datasets) {
    pointers.push(pointer(dataset.inputs.data as Float32Array));
    pointers.push(pointer(dataset.outputs.data as Float32Array));
  }
  return new BigUint64Array(pointers);
}
