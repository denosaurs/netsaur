#include <stdio.h>
#include <include/network.h>
#include <include/matrix.h>

void xor() {
  Network* network = network_create(
    2, // input_size
    COST_CROSS_ENTROPY,
    2, // num_layers
    (Layer*[]) {
      layer_dense(3, ACT_SIGMOID),
      layer_dense(1, ACT_SIGMOID)
    }
  );

  Dataset* dataset = malloc(sizeof(Dataset));
  dataset->inputs = matrix_new_from_array(4, 2, TYPE_F32, (void*) ((float[]) {
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 1.0f
  }));
  dataset->outputs = matrix_new_from_array(4, 1, TYPE_F32, (void*) ((float[]) {
    0.0f,
    1.0f,
    1.0f,
    0.0f
  }));

  network_train(
    network,
    1, // num_datasets
    (Dataset*[]) { dataset },
    5000, // epochs
    0.1f // learning_rate
  );

  Matrix* input = matrix_new_from_array(4, 2, TYPE_F32, (void*) ((float[]) {
    1.0f, 0.0f,
    0.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 1.0f
  }));

  matrix_print(input, "input");
  
  Matrix* output = network_feed_forward(network, input);
  matrix_print(output, "output");
  
  network_free(network);
}

void linear_regression() {
  Network* network = network_create(
    1,
    COST_MEAN_SQUARED_ERROR,
    1,
    (Layer*[]) {
      layer_dense(1, ACT_RELU)
    }
  );

  Dataset* dataset = malloc(sizeof(Dataset));
  dataset->inputs = matrix_new_from_array(4, 1, TYPE_F32, (void*) ((float[]) {
    1.0f,
    2.0f,
    3.0f,
    4.0f
  }));
  dataset->outputs = matrix_new_from_array(4, 1, TYPE_F32, (void*) ((float[]) {
    1.0f,
    3.0f,
    5.0f,
    7.0f
  }));

  matrix_print(network->layers[0]->weights, "weights");
  matrix_print(network->layers[0]->biases, "biases");

  network_train(
    network,
    1, // num_datasets
    (Dataset*[]) { dataset },
    5000, // epochs
    0.1f // learning_rate
  );

  matrix_print(network->layers[0]->weights, "weights");
  matrix_print(network->layers[0]->biases, "biases");

  Matrix* input = matrix_new_from_array(1, 1, TYPE_F32, (void*) ((float[]) {
    5.0f
  }));
  
  // matrix_print(input, "input");

  Matrix* output = network_feed_forward(network, input);
  matrix_print(output, "output");

  network_free(network);
}

int main() {
  printf("XOR\n");
  // store initial time
  clock_t start = clock();
  xor();
  // calculate elapsed time
  clock_t end = clock();
  double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
  printf("Elapsed time: %f ms\n", elapsed * 1000);
  // printf("Linear Regression\n");
  // linear_regression();
  return 0;
}
