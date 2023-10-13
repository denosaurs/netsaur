use crate::DecayScheduler;

pub struct CPUDecayScheduler {
    pub rate: f32,
    pub step_size: usize,
}

impl CPUDecayScheduler {
    pub fn new(config: &DecayScheduler) -> Self {
        CPUDecayScheduler {
            rate: config.rate,
            step_size: config.step_size,
        }
    }
    pub fn exponential(&self, rate: f32, step: usize) -> f32 {
        rate * self.rate.powi((step / self.step_size) as i32)
    }
    pub fn linear(&self, rate: f32, step: usize) -> f32 {
        rate / (self.rate * (1 + step / self.step_size) as f32)
    }
}
