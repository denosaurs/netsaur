use ndarray::ArrayD;
use crate::PostProcessor;

mod step;
use step::CPUStepFunction;

pub enum CPUPostProcessor {
    None,
    Sign,
    Step(CPUStepFunction),
}

impl CPUPostProcessor {
    pub fn from(processor: &PostProcessor) -> Self {
        match processor {
            PostProcessor::None => CPUPostProcessor::None,
            PostProcessor::Sign => CPUPostProcessor::Sign,
            PostProcessor::Step(config) => CPUPostProcessor::Step(CPUStepFunction::new(config)),
        }
    }
    pub fn process(&self, x: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            CPUPostProcessor::None => x,
            CPUPostProcessor::Sign => x.map(|y| y.signum()),
            CPUPostProcessor::Step(processor) => x.map(|y| processor.step(*y)),
        }
    }
}