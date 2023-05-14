import {
  Cost,
  CPU,
  DenseLayer,
  Sequential,
  setupBackend,
  SigmoidLayer,
  tensor2D,
} from "../../mod.ts";

/**
 * Setup the CPU backend. This backend is fast but doesn't work on the Edge.
 */
await setupBackend(CPU);

/**
 * Creates a sequential neural network.
 */
const net = new Sequential({
  /**
   * The number of minibatches is set to 4 and the output size is set to 2.
   */
  size: [4, 2],

  /**
   * The silent option is set to true, which means that the network will not output any logs during trainin
   */
  silent: true,

  /**
   * Defines the layers of a neural network in the XOR function example.
   * The neural network has two input neurons and one output neuron.
   * The layers are defined as follows:
   * - A dense layer with 3 neurons.
   * - sigmoid activation layer.
   * - A dense layer with 1 neuron.
   * -A sigmoid activation layer.
   */
  layers: [
    DenseLayer({ size: [3] }),
    SigmoidLayer(),
    DenseLayer({ size: [1] }),
    SigmoidLayer(),
  ],

  /**
   * The cost function used for training the network is the mean squared error (MSE).
   */
  cost: Cost.MSE,
});

const time = performance.now();

/**
 * Train the network on the given data.
 */
net.train(
  [
    {
      inputs: tensor2D([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ]),
      outputs: tensor2D([[0], [1], [1], [0]]),
    },
  ],
  /**
   * The number of iterations is set to 1,000,000.
   */
  1000000,
);

console.log(`training time: ${performance.now() - time}ms`);

net.saveFile("examples/model/xor.test.bin");
