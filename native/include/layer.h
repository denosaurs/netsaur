#pragma once

#include <include/matrix.h>
#include <include/activation.h>

typedef unsigned char LayerType;

#define LAYER_DENSE 0
#define LAYER_CONV2D 1
#define LAYER_MAXPOOL2D 2

typedef struct Layer
{
  LayerType type;

  unsigned int input_size;
  unsigned int output_size;

  ActivationType activation_type;
  const Activation *activation;

  Matrix *input;
  Matrix *output;

  void (*init)(struct Layer *layer, unsigned int input_size);
  void (*reset)(struct Layer *layer, unsigned int batches);
  Matrix *(*feed_forward)(struct Layer *layer, Matrix *input);
  void (*back_prop)(struct Layer *layer, Matrix *error, float learning_rate);

  // Generic layer data
  void *data;
  void (*finalizer)(void *data);
} Layer;

typedef struct DenseLayer
{
  Matrix *weights;
  Matrix *biases;
} DenseLayer;

typedef struct Conv2dLayer
{
  unsigned int padding;
  unsigned int stride;
  Matrix *kernel;
  Matrix *padded;
} Conv2dLayer;

typedef struct MaxPool2dLayer
{
  unsigned int stride;
} MaxPool2dLayer;

void layer_free(Layer *layer);

Layer *layer_dense(unsigned int size, ActivationType activation_type);
void layer_dense_serialize(Layer *layer, FILE *file);
Layer *layer_dense_deserialize(FILE *file);

Layer *layer_conv2d(unsigned int padding, unsigned int stride, Matrix *kernel, ActivationType activation_type);
void layer_conv2d_serialize(Layer *layer, FILE *file);
Layer *layer_conv2d_deserialize(FILE *file);

Layer *layer_max_pool2d(unsigned int stride);
void layer_max_pool2d_serialize(Layer *layer, FILE *file);
Layer *layer_max_pool2d_deserialize(FILE *file);

void layer_serialize(Layer *layer, FILE *file);
Layer *layer_deserialize(FILE *file);
