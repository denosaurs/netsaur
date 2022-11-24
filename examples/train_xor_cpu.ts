import {
  DenseLayer,
  NeuralNetwork,
  setupBackend,
  SigmoidLayer,
  tensor1D,
  tensor2D,
} from "../mod.ts";
import { CPU } from "../backends/cpu/mod.ts";

await setupBackend(CPU);

const net = new NeuralNetwork({
  silent: true,
  layers: [
    DenseLayer({ size: [3] }),
    SigmoidLayer(),
    DenseLayer({ size: [1] }),
    SigmoidLayer(),
  ],
  cost: "mse",
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
  10000,
);

console.log(`training time: ${performance.now() - time}ms`);
console.log((await net.predict(tensor1D([0, 0]))).data);
console.log((await net.predict(tensor1D([1, 0]))).data);
console.log((await net.predict(tensor1D([0, 1]))).data);
console.log((await net.predict(tensor1D([1, 1]))).data);
