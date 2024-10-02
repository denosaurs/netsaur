import type { Shape } from "../../../../tensor/mod.ts";
import type { Rank } from "../../../../tensor/src/types.ts";

/**
 * Train Options Interface.
 */
export interface TrainOptions {
  datasets: number;
  inputShape: Shape<Rank>;
  outputShape: Shape<Rank>;
  epochs: number;
  batches: number;
  rate: number;
}

/**
 * Predict Options Interface.
 */
export interface PredictOptions {
  inputShape: Shape<Rank>;
  outputShape: Shape<Rank>;
}
