import { NeuralNetwork, DenseLayer } from "../mod.ts";
import { Native, Matrix } from "../backends/native.ts";

const start = performance.now();

const network = await new NeuralNetwork({
  input: 1,
  layers: [
    new DenseLayer({ size: 1, activation: "linear" }),
  ],
  cost: "mse",
}).setupBackend(Native);

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
