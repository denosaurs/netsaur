import { Rank, Shape } from "../core/api/shape.ts";

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
