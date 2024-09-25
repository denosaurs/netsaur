use crate::StepFunctionConfig;

pub struct CPUStepFunction {
    thresholds: Vec<f32>,
    values: Vec<f32>
}
impl CPUStepFunction {
    pub fn new(config: &StepFunctionConfig) -> Self {
        return Self {
            thresholds: config.thresholds.clone(),
            values: config.values.clone()
        }
    }
    pub fn step(&self, x: f32) -> f32 {
        for (i, &threshold) in self.thresholds.iter().enumerate() {
            if x < threshold {
                return self.values[i];
            }
        }
        return self.values.last().unwrap().clone()
    }
}