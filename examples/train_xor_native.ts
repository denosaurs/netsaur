import { NeuralNetwork, DenseLayer, tensor2D, tensor1D, setupBackend } from "../mod.ts";
import { Native } from "../backends/native/mod.ts";

await setupBackend(Native);
const start = performance.now();

const network = new NeuralNetwork({
  input: 2,
  layers: [
    DenseLayer({ size: 3, activation: "sigmoid" }),
    DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

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
