import { OptimizerType } from "../types.ts";

export type Optimizer =
  | { type: OptimizerType.SGD }
  | { type: OptimizerType.Adam; config: AdamOptimizerConfig }
  | { type: OptimizerType.Nadam; config: AdamOptimizerConfig }
  | { type: OptimizerType.RMSProp; config: RMSPropOptimizerConfig };

export interface AdamOptimizerConfig {
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

export interface RMSPropOptimizerConfig {
  decayRate?: number;
  epsilon?: number;
}

export function SGDOptimizer(): Optimizer {
  return { type: OptimizerType.SGD };
}

export function AdamOptimizer(config: AdamOptimizerConfig = {}): Optimizer {
  config.beta1 = config.beta1 || 0.9;
  config.beta2 = config.beta2 || 0.999;
  config.epsilon = config.epsilon || 1e-8;
  return { type: OptimizerType.Adam, config };
}

export function NadamOptimizer(config: AdamOptimizerConfig = {}): Optimizer {
  config.beta1 = config.beta1 || 0.9;
  config.beta2 = config.beta2 || 0.999;
  config.epsilon = config.epsilon || 1e-8;
  return { type: OptimizerType.Nadam, config };
}

export function RMSPropOptimizer(
  config: RMSPropOptimizerConfig = {},
): Optimizer {
  config.decayRate = config.decayRate || 0.9;
  config.epsilon = config.epsilon || 1e-8;
  return { type: OptimizerType.RMSProp, config };
}
