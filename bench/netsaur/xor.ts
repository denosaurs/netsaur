import { NeuralNetwork, DenseLayer } from "../../mod.ts";
import { Native } from "../../src/native/mod.ts";
import { Matrix } from "../../src/native/matrix.ts";

const start = Date.now();

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

console.log("training time", Date.now() - start);

console.log(
  network.predict(
    Matrix.of([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]),
  ),
);
