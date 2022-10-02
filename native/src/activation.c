#include <include/activation.h>

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x) {
  return x * (1.0 - x);
}

const Activation Sigmoid = { sigmoid, sigmoid_prime };

float tanh_prime(float x) {
  return 1.0 - x * x;
}

const Activation Tanh = { tanf, tanh_prime };

float relu(float x) {
  return x > 0.0 ? x : 0.0;
}

float relu_prime(float x) {
  return x > 0.0 ? 1.0 : 0.0;
}

const Activation Relu = { relu, relu_prime };

const Activation* get_activation(ActivationType type) {
  switch (type) {
    case ACT_SIGMOID:
      return &Sigmoid;
    case ACT_TANH:
      return &Tanh;
    case ACT_RELU:
      return &Relu;
    default:
      return NULL;
  }
}
