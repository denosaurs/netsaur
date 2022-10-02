#pragma once

#include <math.h>
#include <stddef.h>

typedef struct Activation {
  float (*fx) (float x);
  float (*dfx) (float x);
} Activation;

typedef unsigned char ActivationType;

#define ACT_SIGMOID (ActivationType) 0
#define ACT_TANH (ActivationType) 1
#define ACT_RELU (ActivationType) 2

const Activation* get_activation(ActivationType type);
