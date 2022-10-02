#include <include/layer.h>

void layer_free(Layer* layer) {
  if (layer->finalizer != NULL) {
    layer->finalizer(layer->data);
  }
  if (layer->weights != NULL) matrix_free(layer->weights);
  if (layer->biases != NULL) matrix_free(layer->biases);
  if (layer->output != NULL) matrix_free(layer->output);
  free(layer);
}

void layer_serialize(Layer* layer, FILE* file) {
  fwrite(&layer->type, sizeof(LayerType), 1, file);
  switch (layer->type) {
    case LAYER_DENSE:
      layer_dense_serialize(layer, file);
      break;
  }
}

Layer* layer_deserialize(FILE* file) {
  LayerType type;
  fread(&type, sizeof(LayerType), 1, file);
  switch (type) {
    case LAYER_DENSE:
      return layer_dense_deserialize(file);
  }
  return NULL;
}
