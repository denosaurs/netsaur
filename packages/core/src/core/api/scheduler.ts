import { SchedulerType } from "../types.ts";

export type Scheduler =
  | { type: SchedulerType.None }
  | {
    type: SchedulerType.LinearDecay | SchedulerType.ExponentialDecay;
    config: DecaySchedulerConfig;
  }
  | { type: SchedulerType.OneCycle; config: OneCycleSchedulerConfig };

export type DecaySchedulerConfig = {
  rate?: number;
  step_size?: number;
};

export type OneCycleSchedulerConfig = {
  max_rate?: number;
  step_size?: number;
};

export function NoScheduler(): Scheduler {
  return { type: SchedulerType.None };
}

export function LinearDecay(config: DecaySchedulerConfig = {}): Scheduler {
  config.rate = config.rate || 0.99;
  config.step_size = config.step_size || 100;
  return { type: SchedulerType.LinearDecay, config };
}

export function ExponentialDecay(config: DecaySchedulerConfig = {}): Scheduler {
  config.rate = config.rate || 0.99;
  config.step_size = config.step_size || 100;
  return { type: SchedulerType.ExponentialDecay, config };
}

export function OneCycle(config: OneCycleSchedulerConfig = {}): Scheduler {
  config.max_rate = config.max_rate || 0.01;
  config.step_size = config.step_size || 100;
  return { type: SchedulerType.OneCycle, config };
}
