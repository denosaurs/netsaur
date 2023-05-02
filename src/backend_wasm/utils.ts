import { Rank, Shape } from "../../mod.ts";

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
