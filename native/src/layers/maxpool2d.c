#include <include/layer.h>

void layer_max_pool2d_init(Layer *layer, unsigned int input_size)
{
  // TODO
}

void layer_max_pool2d_reset(Layer *layer, unsigned int batches)
{
  // TODO
}

Matrix *layer_max_pool2d_feed_forward(Layer *layer, Matrix *input)
{
  // TODO
}

void layer_max_pool2d_back_prop(Layer *layer, Matrix *error, float learning_rate)
{
  // TODO
}

void layer_max_pool2d_free(void *data)
{
  free(data);
}

Layer *layer_max_pool2d(unsigned int stride)
{
  Layer *layer = malloc(sizeof(Layer));

  layer->type = LAYER_MAXPOOL2D;

  layer->input_size = 0;  // Set by init
  layer->output_size = 0; // Set by init

  layer->activation_type = ACT_NONE;
  layer->activation = NULL;

  // Set by reset
  layer->output = NULL;

  layer->init = layer_max_pool2d_init;
  layer->reset = layer_max_pool2d_reset;
  layer->feed_forward = layer_max_pool2d_feed_forward;
  layer->back_prop = layer_max_pool2d_back_prop;

  MaxPool2dLayer *pool = malloc(sizeof(MaxPool2dLayer));
  pool->stride = stride;
  layer->data = pool;
  layer->finalizer = layer_max_pool2d_free;

  return layer;
}

void layer_max_pool2d_serialize(Layer *layer, FILE *file)
{
  fwrite(&layer->input_size, sizeof(unsigned int), 1, file);
  fwrite(&layer->output_size, sizeof(unsigned int), 1, file);
  MaxPool2dLayer *pool = (MaxPool2dLayer *)layer->data;
  fwrite(&pool->stride, sizeof(unsigned int), 1, file);
}

Layer *layer_max_pool2d_deserialize(FILE *file)
{
  Layer *layer = malloc(sizeof(Layer));

  fread(&layer->input_size, sizeof(unsigned int), 1, file);
  fread(&layer->output_size, sizeof(unsigned int), 1, file);

  MaxPool2dLayer *pool = malloc(sizeof(MaxPool2dLayer));
  fread(&pool->stride, sizeof(unsigned int), 1, file);

  layer->type = LAYER_MAXPOOL2D;

  layer->activation_type = ACT_NONE;
  layer->activation = NULL;

  // Set by reset
  layer->output = NULL;

  layer->init = layer_max_pool2d_init;
  layer->reset = layer_max_pool2d_reset;
  layer->feed_forward = layer_max_pool2d_feed_forward;
  layer->back_prop = layer_max_pool2d_back_prop;

  layer->data = pool;
  layer->finalizer = layer_max_pool2d_free;

  return layer;
}
