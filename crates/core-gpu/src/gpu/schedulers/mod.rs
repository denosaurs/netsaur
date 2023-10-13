mod decay;
mod oc;

use crate::Scheduler;

pub use decay::*;
pub use oc::*;
pub enum GPUScheduler {
    None,
    LinearDecay(GPUDecayScheduler),
    ExponentialDecay(GPUDecayScheduler),
    OneCycle(GPUOneCycleScheduler),
}

impl GPUScheduler {
    pub fn from(scheduler: &Scheduler) -> Self {
        match scheduler {
            Scheduler::None => GPUScheduler::None,
            Scheduler::LinearDecay(config) => {
                GPUScheduler::LinearDecay(GPUDecayScheduler::new(config))
            }
            Scheduler::ExponentialDecay(config) => {
                GPUScheduler::ExponentialDecay(GPUDecayScheduler::new(config))
            }
            Scheduler::OneCycle(config) => {
                GPUScheduler::OneCycle(GPUOneCycleScheduler::new(config))
            }
        }
    }
    pub fn eta(&self, rate: f32, step: usize) -> f32 {
        match self {
            GPUScheduler::None => rate,
            GPUScheduler::LinearDecay(scheduler) => scheduler.linear(rate, step),
            GPUScheduler::ExponentialDecay(scheduler) => scheduler.exponential(rate, step),
            GPUScheduler::OneCycle(scheduler) => scheduler.eta(rate, step),
        }
    }
}
