#pragma once

#include <stddef.h>
#include <math.h>

typedef struct Cost {
  float (*fx) (float* y, float* y_hat, unsigned int n);
  float (*dfx) (float y, float y_hat);
} Cost;

typedef unsigned char CostType;

#define COST_MEAN_SQUARED_ERROR (CostType) 0
#define COST_CROSS_ENTROPY (CostType) 1

const Cost* get_cost(CostType type);
