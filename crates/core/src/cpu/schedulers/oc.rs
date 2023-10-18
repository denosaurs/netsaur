use crate::OneCycleScheduler;

pub struct CPUOneCycleScheduler {
    pub max_rate: f32,
    pub step_size: usize,
}

impl CPUOneCycleScheduler {
    pub fn new(config: &OneCycleScheduler) -> Self {
        CPUOneCycleScheduler {
            max_rate: config.max_rate,
            step_size: config.step_size,
        }
    }
    pub fn eta(&self, rate: f32, step: usize) -> f32 {
        let steps = self.step_size as f32;
        let step = step % (2 * self.step_size);
        if step < self.step_size {
            rate + (self.max_rate - rate) * (step as f32) / (steps)
        } else {
            self.max_rate - (self.max_rate - rate) * ((step - self.step_size) as f32) / (steps)
        }
    }
}
