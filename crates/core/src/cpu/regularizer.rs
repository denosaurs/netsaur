use ndarray::ArrayD;

pub struct CPURegularizer {
    l1_strength: f32,
    l2_strength: f32,
}

impl CPURegularizer {
    pub fn from(c: f32, l1_ratio: f32) -> Self {
        if c == 0.0 {
            return CPURegularizer {
                l1_strength: 0.0,
                l2_strength: 0.0
            }
        }
        let strength = 1.0 / c;
        if l1_ratio == 1.0 {
            CPURegularizer {
                l1_strength: strength,
                l2_strength: 0.0,
            }
        } else if l1_ratio == 0.0 {
            CPURegularizer {
                l1_strength: 0.0,
                l2_strength: strength,
            }
        } else {
            let l1_strength = strength * l1_ratio;
            let l2_strength = strength - l1_strength;
            CPURegularizer {
                l1_strength,
                l2_strength,
            }
        }
    }
    pub fn l1_coeff(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        if self.l1_strength == 0.0 {
            ArrayD::zeros(x.shape())
        } else {
            self.l1_strength * x.map(|w| w.abs())
        }
    }
    pub fn l2_coeff(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        if self.l2_strength == 0.0 {
            ArrayD::zeros(x.shape())
        } else {
            self.l2_strength * x.map(|w| w * w)
        }
    }
    pub fn coeff(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        self.l1_coeff(x) + self.l2_coeff(x)
    }
}
