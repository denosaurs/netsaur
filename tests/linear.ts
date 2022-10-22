import { DenseLayer, NeuralNetwork, setupBackend } from "../mod.ts";
import { Matrix, Native } from "../backends/native/mod.ts";

const start = performance.now();

await setupBackend(Native);

const network = new NeuralNetwork({
  input: 1,
  layers: [
    DenseLayer({ size: 1, activation: "linear" }),
  ],
  cost: "mse",
});

network.train(
  [
    {
      // y = 2x + 1
      inputs: Matrix.column([1, 2, 3, 4]),
      outputs: Matrix.column([3, 5, 7, 9]),
    },
  ],
  500,
  0.01,
);

console.log("training time", performance.now() - start);
const prediction = await network.predict(
  Matrix.column([20]),
);

console.log(
  prediction.data.map((e: number) => Math.round(e)),
);
