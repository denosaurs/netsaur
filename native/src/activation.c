#include <include/activation.h>

float sigmoid(float x)
{
  return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x, float error)
{
  return x * (1.0 - x) * error;
}

const Activation Sigmoid = {sigmoid, sigmoid_prime};

float tanh_prime(float x, float error)
{
  return (1.0 - (x * x)) * error;
}

const Activation Tanh = {tanf, tanh_prime};

float relu(float x)
{
  return x > 0.0 ? x : 0.0;
}

float relu_prime(float x, float error)
{
  return x > 0.0 ? error : 0.0;
}

const Activation Relu = {relu, relu_prime};

float linear(float x)
{
  return x;
}

float linear_prime(float x, float error)
{
  return error;
}

const Activation Linear = {linear, linear_prime};

float leaky_relu(float x)
{
  return x > 0.0 ? x : 0.01 * x;
}

float leaky_relu_prime(float x, float error)
{
  return x > 0.0 ? error : 0.01 * error;
}

const Activation LeakyRelu = {leaky_relu, leaky_relu_prime};

float relu6(float x)
{
  return x > 0.0 ? (x < 6.0 ? x : 6.0) : 0.0;
}

float relu6_prime(float x, float error)
{
  return x > 0.0 ? (x < 6.0 ? error : 0.0) : 0.0;
}

const Activation Relu6 = {relu6, relu6_prime};

float elu(float x)
{
  return x > 0.0 ? x : 0.01 * (exp(x) - 1.0);
}

float elu_prime(float x, float error)
{
  return x > 0.0 ? error : 0.01 * error * exp(x);
}

const Activation Elu = {elu, elu_prime};

float selu(float x)
{
  return x > 0.0 ? 1.0507 * x : 1.0507 * 1.67326 * (exp(x) - 1.0);
}

float selu_prime(float x, float error)
{
  return x > 0.0 ? 1.0507 * error : 1.0507 * 1.67326 * error * exp(x);
}

const Activation Selu = {selu, selu_prime};

const Activation *get_activation(ActivationType type)
{
  switch (type)
  {
  case ACT_SIGMOID:
    return &Sigmoid;
  case ACT_TANH:
    return &Tanh;
  case ACT_RELU:
    return &Relu;
  case ACT_LINEAR:
    return &Linear;
  case ACT_LEAKY_RELU:
    return &LeakyRelu;
  case ACT_RELU6:
    return &Relu6;
  case ACT_ELU:
    return &Elu;
  case ACT_SELU:
    return &Selu;
  default:
    return NULL;
  }
}
