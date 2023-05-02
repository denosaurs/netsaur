import { DenseLayer, NeuralNetwork, tensor1D, tensor2D } from "../../mod.ts";
import { Model } from "../../model/mod.ts";

const net = new NeuralNetwork({
  silent: true,
  layers: [
    DenseLayer({ size: 3, activation: "sigmoid" }),
    DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

const time = performance.now();

await net.train(
  [
    {
      inputs: await tensor2D([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ]),
      outputs: await tensor1D([0, 1, 1, 0]),
    },
  ],
  5000,
  4,
  0.1,
);

console.log(`training time: ${performance.now() - time}ms`);

await Model.save("./examples/train_and_run/network.json", net);
