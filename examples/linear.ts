/**
 * This example trains a neural network to predict the output of the function y = 2x + 1
 */

import {
  Cost,
  CPU,
  DenseLayer,
  Sequential,
  setupBackend,
  tensor2D,
} from "../mod.ts";

/**
 * The test data used for predicting the output of the function y = 2x + 1
 */
const testData = [20, 40, 43, 87, 43];

/**
 * Setup the CPU backend. This backend is fast but doesn't work on the Edge.
 */
await setupBackend(CPU);

/**
 * Creates a sequential neural network.
 */
const network = new Sequential({
  /**
   * The number of minibatches is set to 4 and the output size is set to 1.
   */
  size: [4, 1],

  /**
   * The silent option is set to true, which means that the network will not output any logs during trainin
   */
  silent: true,

  /**
   * Creates two dense layers, with the first layer having 3 neurons and the second layer having 1 neuron.
   */
  layers: [DenseLayer({ size: [3] }), DenseLayer({ size: [1] })],

  /**
   * The cost function used for training the network is the mean squared error (MSE).
   */
  cost: Cost.MSE,
});

const start = performance.now();

/**
 * Train the network on the given data.
 */
network.train(
  [
    {
      // y = 2x + 1
      inputs: tensor2D([[1], [2], [3], [4]]),
      outputs: tensor2D([[3], [5], [7], [9]]),
    },
  ],
  /**
   * The number of iterations is set to 400.
   */
  400,
  /**
   * The number of batches is set to 1.
   */
  1,
  /**
   * The learning rate is set to 0.01.
   */
  0.01,
);

console.log("training time", performance.now() - start, " milliseconds");
console.log("y = 2x + 1");

/**
 * Make a prediction on the test data.
 */
const predicted = await network.predict(tensor2D(testData.map((x) => [x])));
for (const [i, res] of predicted.data.entries()) {
  console.log(
    `input: ${testData[i]}\noutput: ${res.toFixed(2)}\nexpected: ${
      2 * testData[i] + 1
    }\n`,
  );
}
