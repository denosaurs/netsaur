import type { Rank, Shape } from "../../core/api/shape.ts";

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
