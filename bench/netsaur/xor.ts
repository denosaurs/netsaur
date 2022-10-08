import { NeuralNetwork, DenseLayer } from "../../mod.ts";
import { Native, Matrix  } from "../../backends/native.ts";

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
  0.1,
);

const training_time = performance.now() - start;
console.log("training time", training_time);

console.log(
  await network.predict(
    Matrix.of([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]),
  ),
);
