import { DenseLayer, NeuralNetwork, tensor2D, tensor1D } from "../mod.ts";
import { CPU } from "../backends/cpu/mod.ts";

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    new DenseLayer({ size: 3, activation: "sigmoid" }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
}).setupBackend(CPU);

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
      outputs:  await tensor1D([0, 1, 1, 0]),
    },
  ],
  5000,
);

console.log(`training time: ${performance.now() - time}ms`);
console.log(await net.predict(new Float32Array([0, 0])));
console.log(await net.predict(new Float32Array([1, 0])));
console.log(await net.predict(new Float32Array([0, 1])));
console.log(await net.predict(new Float32Array([1, 1])));
