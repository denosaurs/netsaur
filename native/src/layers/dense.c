#include <include/layer.h>

void layer_dense_init(Layer *layer, unsigned int input_size)
{
  layer->input_size = input_size;
  DenseLayer *dense = (DenseLayer *)layer->data;
  dense->weights = matrix_new_randf(input_size, layer->output_size);
  dense->biases = matrix_new(1, layer->output_size, TYPE_F32);
}

void layer_dense_reset(Layer *layer, unsigned int batches)
{
  if (layer->output != NULL && layer->output->returned_to_js != 1)
    matrix_free(layer->output);
  layer->output = matrix_new(batches, layer->output_size, TYPE_F32);
}

Matrix *layer_dense_feed_forward(Layer *layer, Matrix *input)
{
  DenseLayer *dense = layer->data;

  layer->input = input;
  // matrix_print(input, "input");
  // matrix_print(dense->weights, "weights");
  matrix_dot(input, dense->weights, layer->output);
  float *output_data = (float *)layer->output->data;
  float *bias_data = (float *)dense->biases->data;
  for (int i = 0, j = 0; i < layer->output->rows * layer->output->cols; i++, j++)
  {
    if (j == layer->output->cols)
    {
      j = 0;
    }
    output_data[i] = layer->activation->fx(output_data[i] + bias_data[j]);
  }
  return layer->output;
}

void layer_dense_back_prop(Layer *layer, Matrix *error, float learning_rate)
{
  DenseLayer *dense = (DenseLayer *)layer->data;

  Matrix *cost = matrix_new(error->rows, error->cols, error->type);

  float *cost_data = (float *)cost->data;
  float *error_data = (float *)error->data;
  float *output_data = (float *)layer->output->data;
  for (int i = 0; i < error->rows * error->cols; i++)
  {
    cost_data[i] = layer->activation->dfx(output_data[i], error_data[i]);
  }

  Matrix *input_transpose = matrix_transpose(layer->input);
  Matrix *weights_delta = matrix_dot(input_transpose, cost, NULL);
  matrix_free(input_transpose);

  float *weights_delta_data = (float *)weights_delta->data;
  float *weights_data = (float *)dense->weights->data;
  for (int i = 0; i < weights_delta->rows * weights_delta->cols; i++)
  {
    weights_data[i] += weights_delta_data[i] * learning_rate;
  }

  matrix_free(weights_delta);

  float *biases_data = (float *)dense->biases->data;
  for (int i = 0, j = 0; i < error->rows * error->cols; i++, j++)
  {
    if (j >= dense->biases->cols)
    {
      j = 0;
    }
    biases_data[j] += cost_data[i] * learning_rate;
  }

  matrix_free(cost);
}

void layer_dense_free(void *data)
{
  DenseLayer *dense = (DenseLayer *)data;
  matrix_free(dense->weights);
  matrix_free(dense->biases);
  free(dense);
}

Layer *layer_dense(unsigned int size, ActivationType activation_type)
{
  Layer *layer = malloc(sizeof(Layer));

  layer->type = LAYER_DENSE;

  layer->input_size = 0; // Set by init
  layer->output_size = size;

  layer->activation_type = activation_type;
  layer->activation = get_activation(activation_type);

  // Set by reset
  layer->output = NULL;

  layer->init = layer_dense_init;
  layer->reset = layer_dense_reset;
  layer->feed_forward = layer_dense_feed_forward;
  layer->back_prop = layer_dense_back_prop;

  layer->data = malloc(sizeof(DenseLayer));
  layer->finalizer = layer_dense_free;

  return layer;
}

void layer_dense_serialize(Layer *layer, FILE *file)
{
  fwrite(&layer->input_size, sizeof(unsigned int), 1, file);
  fwrite(&layer->output_size, sizeof(unsigned int), 1, file);
  fwrite(&layer->activation_type, sizeof(ActivationType), 1, file);
  DenseLayer *dense = (DenseLayer *)layer->data;
  matrix_serialize(dense->weights, file);
  matrix_serialize(dense->biases, file);
}

Layer *layer_dense_deserialize(FILE *file)
{
  Layer *layer = malloc(sizeof(Layer));

  fread(&layer->input_size, sizeof(unsigned int), 1, file);
  fread(&layer->output_size, sizeof(unsigned int), 1, file);
  fread(&layer->activation_type, sizeof(ActivationType), 1, file);

  DenseLayer *dense = malloc(sizeof(DenseLayer));

  dense->weights = matrix_deserialize(file);
  dense->biases = matrix_deserialize(file);

  layer->type = LAYER_DENSE;

  layer->activation = get_activation(layer->activation_type);

  // Set by reset
  layer->output = NULL;

  layer->init = layer_dense_init;
  layer->reset = layer_dense_reset;
  layer->feed_forward = layer_dense_feed_forward;
  layer->back_prop = layer_dense_back_prop;

  layer->data = dense;
  layer->finalizer = layer_dense_free;

  return layer;
}
