import { NeuralNetwork, DenseLayer, Tensor, tensor2D, tensor1D } from "../mod.ts";
import { Native } from "../backends/native/mod.ts";

Tensor.setupBackend(Native);
const start = performance.now();

const network = await new NeuralNetwork({
  input: 2,
  layers: [
    new DenseLayer({ size: 3, activation: "sigmoid" }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
}).setupBackend(Native);

network.train(
  [
    {
      inputs: await tensor2D([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]),
      outputs: await tensor1D([0, 1, 1, 0]),
    },
  ],
  5000,
  0.1,
);

console.log("training time", performance.now() - start, "milliseconds");

console.log(
  await network.predict(
    await tensor2D([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]),
  ),
);
