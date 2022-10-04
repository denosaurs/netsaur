#include <include/layer.h>

void layer_conv2d_init(Layer *layer, unsigned int input_size)
{
  // TODO
}

void layer_conv2d_reset(Layer *layer, unsigned int batches)
{
  // TODO
}

Matrix *layer_conv2d_feed_forward(Layer *layer, Matrix *input)
{
  // TODO
}

void layer_conv2d_back_prop(Layer *layer, Matrix *error, float learning_rate)
{
  // TODO
}

void layer_conv2d_free(void *data)
{
  Conv2dLayer *conv = (Conv2dLayer *)data;
  matrix_free(conv->kernel);
  if (conv->padded != NULL)
    matrix_free(conv->padded);
  free(data);
}

Layer *layer_conv2d(unsigned int padding, unsigned int stride, Matrix *kernel, ActivationType activation_type)
{
  Layer *layer = malloc(sizeof(Layer));

  layer->type = LAYER_CONV2D;

  layer->input_size = 0;  // Set by init
  layer->output_size = 0; // Set by init

  layer->activation_type = activation_type;
  layer->activation = get_activation(activation_type);

  // Set by reset
  layer->output = NULL;

  layer->init = layer_conv2d_init;
  layer->reset = layer_conv2d_reset;
  layer->feed_forward = layer_conv2d_feed_forward;
  layer->back_prop = layer_conv2d_back_prop;

  Conv2dLayer *conv = malloc(sizeof(Conv2dLayer));
  conv->padding = padding;
  conv->stride = stride;
  conv->kernel = kernel;
  conv->padded = NULL;
  layer->data = conv;
  layer->finalizer = layer_conv2d_free;

  return layer;
}

void layer_conv2d_serialize(Layer *layer, FILE *file)
{
  fwrite(&layer->input_size, sizeof(unsigned int), 1, file);
  fwrite(&layer->output_size, sizeof(unsigned int), 1, file);
  fwrite(&layer->activation_type, sizeof(ActivationType), 1, file);
  Conv2dLayer *conv = (Conv2dLayer *)layer->data;
  fwrite(&conv->stride, sizeof(unsigned int), 1, file);
  fwrite(&conv->padding, sizeof(unsigned int), 1, file);
  matrix_serialize(conv->kernel, file);
}

Layer *layer_conv2d_deserialize(FILE *file)
{
  Layer *layer = malloc(sizeof(Layer));

  fread(&layer->input_size, sizeof(unsigned int), 1, file);
  fread(&layer->output_size, sizeof(unsigned int), 1, file);
  fread(&layer->activation_type, sizeof(ActivationType), 1, file);

  Conv2dLayer *conv = malloc(sizeof(Conv2dLayer));
  fread(&conv->stride, sizeof(unsigned int), 1, file);
  fread(&conv->padding, sizeof(unsigned int), 1, file);
  conv->kernel = matrix_deserialize(file);
  conv->padded = NULL;

  layer->type = LAYER_CONV2D;

  layer->activation = get_activation(layer->activation_type);

  // Set by reset
  layer->output = NULL;

  layer->init = layer_conv2d_init;
  layer->reset = layer_conv2d_reset;
  layer->feed_forward = layer_conv2d_feed_forward;
  layer->back_prop = layer_conv2d_back_prop;

  layer->data = conv;
  layer->finalizer = layer_conv2d_free;

  return layer;
}
