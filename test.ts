import { Layer, Matrix, Network } from "./src/native/mod.ts";

const start = Date.now();

{
  const network = new Network({
    inputSize: 2,
    layers: [
      Layer.dense({ units: 3, activation: "sigmoid" }),
      Layer.dense({ units: 1, activation: "sigmoid" }),
    ],
    cost: "crossentropy",
  });

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

  network.save("xor_model.bin");
}

const network = Network.load("xor_model.bin");

console.log(
  network.predict(
    Matrix.of([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]),
  ).data.map((e) => Math.round(e)),
);

console.log(Date.now() - start);
