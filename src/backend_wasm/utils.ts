import { Rank, Shape } from "../../mod.ts";

/**
 * Train Options Interface.
 */
export type TrainOptions = {
  datasets: number;
  inputShape: Shape[Rank];
  outputShape: Shape[Rank];
  epochs: number;
  rate: number;
};

/**
 * Predict Options Interface.
 */
export type PredictOptions = {
  inputShape: Shape[Rank];
  outputShape: Shape[Rank];
};
