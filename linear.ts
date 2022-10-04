import { Layer, Matrix, Network } from "./src/native/mod.ts";

const start = performance.now();

const network = new Network({
  inputSize: 1,
  layers: [
    Layer.dense({ units: 1, activation: "linear" }),
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

console.log(
  network.predict(
    Matrix.column([20]),
  ).data.map((e) => Math.round(e)),
);
