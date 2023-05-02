import { Rank, Shape } from "../core/api/shape.ts";
import { NeuralNetwork } from "../../mod.ts";

export interface ModelFormat {
  load: (path: string) => Promise<NetworkJSON>;
  save: (path: string, net: NeuralNetwork) => Promise<void>;
}

export type KerasLayerType = "InputLayer" | "Conv2D";

export interface KerasLayerConfig {
  name?: string;
  trainable?: boolean;
  filters?: number;
  // deno-lint-ignore no-explicit-any
  kernel_size?: any[];
  strides?: number[];
  padding?: string;
  data_format?: string;
  // deno-lint-ignore no-explicit-any
  dilation_rate?: any[];
  activation?: string;
  use_bias?: boolean;
  // deno-lint-ignore no-explicit-any
  kernel_initializer?: any;
  // deno-lint-ignore no-explicit-any
  bias_initializer?: any;
  // deno-lint-ignore no-explicit-any
  kernel_regularizer?: any;
  // deno-lint-ignore no-explicit-any
  bias_regularizer?: any;
  // deno-lint-ignore no-explicit-any
  activity_regularizer?: any;
  // deno-lint-ignore no-explicit-any
  kernel_constraints?: any;
  // deno-lint-ignore no-explicit-any
  bias_constraints?: any;
  // deno-lint-ignore no-explicit-any
  batch_input_shape?: any[];
  dtype?: string;
  sparse?: boolean;
}
export interface KerasLayer {
  name: string;
  class_name: KerasLayerType;
  config: KerasLayerConfig;
  // deno-lint-ignore no-explicit-any
  inbound_nodes: any[];
}

export interface LayerJSON {
  outputSize?: Shape[Rank];
  inputSize?: Shape[Rank];
  activationFn?: string;
  costFn?: string;
  type: string;
  weights?: TensorJSON;
  paddedSize?: Shape[Rank];
  biases?: TensorJSON;
  kernel?: TensorJSON;
  strides?: Shape[Rank];
  padding?: number;
  mode?: "max" | "avg";
}

export interface NetworkJSON {
  costFn?: string;
  input: Shape[Rank] | undefined;
  layers: LayerJSON[];
}

export interface TensorJSON {
  data: number[];
  shape: Shape[Rank];
}
