mod decay;
mod oc;

use crate::Scheduler;

pub use decay::*;
pub use oc::*;
pub enum CPUScheduler {
    None,
    LinearDecay(CPUDecayScheduler),
    ExponentialDecay(CPUDecayScheduler),
    OneCycle(CPUOneCycleScheduler)
}

impl CPUScheduler {
    pub fn from(scheduler: &Scheduler) -> Self {
        match scheduler {
            Scheduler::None => CPUScheduler::None,
            Scheduler::LinearDecay(config) => {
                CPUScheduler::LinearDecay(CPUDecayScheduler::new(config))
            }
            Scheduler::ExponentialDecay(config) => {
                CPUScheduler::ExponentialDecay(CPUDecayScheduler::new(config))
            }
            Scheduler::OneCycle(config) => {
                CPUScheduler::OneCycle(CPUOneCycleScheduler::new(config))
            }
        }
    }
    pub fn eta(&self, rate: f32, step: usize) -> f32 {
        match self {
            CPUScheduler::None => rate,
            CPUScheduler::LinearDecay(scheduler) => scheduler.linear(rate, step),
            CPUScheduler::ExponentialDecay(scheduler) => scheduler.exponential(rate, step),
            CPUScheduler::OneCycle(scheduler) => scheduler.eta(rate, step)
        }
    }
}