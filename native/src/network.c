#include <include/network.h>

Network* network_create(
  unsigned int input_size,
  CostType cost_type,
  unsigned int num_layers,
  Layer** layers
) {
  Network* network = malloc(sizeof(Network));

  network->input_size = input_size;
  network->cost_type = cost_type;
  network->cost = get_cost(cost_type);
  network->num_layers = num_layers;
  network->layers = malloc(sizeof(Layer*) * num_layers);
  memcpy(network->layers, layers, sizeof(Layer*) * num_layers);

  for (unsigned int i = 0; i < network->num_layers; i++) {
    network->layers[i]->init(network->layers[i], input_size);
    input_size = network->layers[i]->output_size;
  }

  return network;
}

void* network_free(Network* network) {
  for (unsigned int i = 0; i < network->num_layers; i++) {
    layer_free(network->layers[i]);
  }
  // TODO: why is this invalid
  // free(network->layers);
  free(network);
}

Matrix* network_feed_forward(Network* network, Matrix* input) {
  for (unsigned int i = 0; i < network->num_layers; i++) {
    network->layers[i]->reset(network->layers[i], input->rows);
  }
  Matrix* output = input;
  for (unsigned int i = 0; i < network->num_layers; i++) {
    output = network->layers[i]->feed_forward(network->layers[i], output);
  }
  output->returned_to_js = 1;
  return output;
}

void network_back_prop(Network* network, Matrix* target, float learning_rate) {
  Layer* output_layer = network->layers[network->num_layers - 1];
  Matrix* error = matrix_new(output_layer->output->rows, output_layer->output->cols, output_layer->output->type);
  
  float* error_data = (float*) error->data;
  float* output_layer_data = (float*) output_layer->output->data;
  float* target_data = (float*) target->data;
  for (int i = 0; i < output_layer->output->rows * output_layer->output->cols; i++) {
    error_data[i] = network->cost->dfx(output_layer_data[i], target_data[i]);
  }

  output_layer->back_prop(output_layer, error, learning_rate);

  Matrix* weights = output_layer->weights;
  for (int i = network->num_layers - 2; i >= 0; i--) {
    Layer* layer = network->layers[i];
    Matrix* error_old = error;
    Matrix* weights_transpose = matrix_transpose(weights);
    error = matrix_dot(error, weights_transpose, NULL);
    matrix_free(error_old);
    matrix_free(weights_transpose);
    layer->back_prop(layer, error, learning_rate);
    weights = layer->weights;
  }
  matrix_free(error);
}

void network_train(Network* network, unsigned int num_datasets, Dataset** datasets, unsigned int epochs, float learning_rate) {
  clock_t start = clock();
  for (unsigned int i = 0; i < epochs; i++) {
    printf("Epoch %d\n", i + 1);
    for (unsigned int j = 0; j < num_datasets; j++) {
      Dataset* dataset = datasets[j];
      for (unsigned int i = 0; i < network->num_layers; i++) {
        network->layers[i]->reset(network->layers[i], dataset->inputs->rows);
      }
      network_feed_forward(network, dataset->inputs);
      network_back_prop(network, dataset->outputs, learning_rate);
    }
  }
  clock_t end = clock();
  float elapsed = (float) (end - start) / CLOCKS_PER_SEC * 1000;
  printf("Completed %d epochs in %dms\n", epochs, (unsigned int) elapsed);
}

void network_save(Network* network, const char* filename) {
  FILE* file = fopen(filename, "w");
  fwrite(&network->input_size, sizeof(unsigned int), 1, file);
  fwrite(&network->num_layers, sizeof(unsigned int), 1, file);
  fwrite(&network->cost_type, sizeof(CostType), 1, file);
  for (unsigned int i = 0; i < network->num_layers; i++) {
    Layer* layer = network->layers[i];
    layer_serialize(layer, file);
  }
  fclose(file);
}

Network* network_load(const char* filename) {
  FILE* file = fopen(filename, "r");
  unsigned int input_size;
  unsigned int num_layers;
  CostType cost_type;
  fread(&input_size, sizeof(unsigned int), 1, file);
  fread(&num_layers, sizeof(unsigned int), 1, file);
  fread(&cost_type, sizeof(CostType), 1, file);
  Layer** layers = malloc(sizeof(Layer*) * num_layers);
  for (unsigned int i = 0; i < num_layers; i++) {
    layers[i] = layer_deserialize(file);
  }
  Network* net = malloc(sizeof(Network));
  net->input_size = input_size;
  net->num_layers = num_layers;
  net->cost_type = cost_type;
  net->cost = get_cost(cost_type);
  net->layers = layers;
  fclose(file);
  return net;
}
