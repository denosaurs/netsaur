import {
  DenseLayer,
  NeuralNetwork,
  setupBackend,
  tensor1D,
  tensor2D,
} from "../mod.ts";
import { CPU } from "../backends/cpu/mod.ts";

await setupBackend(CPU);

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
);

console.log(`training time: ${performance.now() - time}ms`);
console.log(await net.predict(new Float32Array([0, 0])));
console.log(await net.predict(new Float32Array([1, 0])));
console.log(await net.predict(new Float32Array([0, 1])));
console.log(await net.predict(new Float32Array([1, 1])));
