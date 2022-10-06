#include <include/layer.h>

void layer_free(Layer *layer)
{
  if (layer->finalizer != NULL)
  {
    layer->finalizer(layer->data);
  }
  if (layer->output != NULL)
    matrix_free(layer->output);
  free(layer);
}

void layer_serialize(Layer *layer, FILE *file)
{
  fwrite(&layer->type, sizeof(LayerType), 1, file);
  switch (layer->type)
  {
  case LAYER_DENSE:
    layer_dense_serialize(layer, file);
    break;
  case LAYER_CONV2D:
    layer_conv2d_serialize(layer, file);
    break;
  case LAYER_MAXPOOL2D:
    layer_max_pool2d_serialize(layer, file);
    break;
  }
}

Layer *layer_deserialize(FILE *file)
{
  LayerType type;
  fread(&type, sizeof(LayerType), 1, file);
  switch (type)
  {
  case LAYER_DENSE:
    return layer_dense_deserialize(file);
  case LAYER_CONV2D:
    return layer_conv2d_deserialize(file);
  case LAYER_MAXPOOL2D:
    return layer_max_pool2d_deserialize(file);
  }
  return NULL;
}
