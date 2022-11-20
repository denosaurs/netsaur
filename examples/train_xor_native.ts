import { NeuralNetwork, DenseLayer, tensor2D, tensor1D, setupBackend } from "../mod.ts";
import { Native } from "../backends/native/mod.ts";

await setupBackend(Native);

const net = new NeuralNetwork({
  silent: true,
  layers: [
    DenseLayer({ size: [3], activation: "sigmoid" }),
    DenseLayer({ size: [1], activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

const time = performance.now();

await net.train(
  [
    {
      inputs: tensor2D([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ]),
      outputs: tensor1D([0, 1, 1, 0]),
    },
  ],
  5000,
);

console.log(`training time: ${performance.now() - time}ms`);
console.log((await net.predict(tensor1D([0, 0]))).data);
console.log((await net.predict(tensor1D([1, 0]))).data);
console.log((await net.predict(tensor1D([0, 1]))).data);
console.log((await net.predict(tensor1D([1, 1]))).data);
