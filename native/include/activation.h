#pragma once

#include <math.h>
#include <stddef.h>

typedef struct Activation
{
  float (*fx)(float x);
  float (*dfx)(float x, float error);
} Activation;

typedef char ActivationType;

#define ACT_NONE (ActivationType) - 1
#define ACT_SIGMOID (ActivationType)0
#define ACT_TANH (ActivationType)1
#define ACT_RELU (ActivationType)2
#define ACT_LINEAR (ActivationType)3
#define ACT_LEAKY_RELU (ActivationType)4
#define ACT_RELU6 (ActivationType)5
#define ACT_ELU (ActivationType)6
#define ACT_SELU (ActivationType)7

const Activation *get_activation(ActivationType type);
