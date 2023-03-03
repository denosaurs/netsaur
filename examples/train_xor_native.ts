import {
  DenseLayer,
  NeuralNetwork,
  setupBackend,
} from "../mod.ts";
import { Matrix, Native } from "../backends/native/mod.ts";

await setupBackend(Native);

const net = new NeuralNetwork({
  silent: true,
  input: [2, 1],
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
      inputs: Matrix.of([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]),
      outputs: Matrix.column([0, 1, 1, 0]),
    },
  ],
  5000,
);

console.log(`training time: ${performance.now() - time}ms`);
console.log(
  await net.predict(
    Matrix.of([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]),
  ),
);