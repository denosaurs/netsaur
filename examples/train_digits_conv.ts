import { CPU } from "../backends/cpu/mod.ts";
import { setupBackend } from "../core/mod.ts";
import {
  ConvLayer,
  DenseLayer,
  NeuralNetwork,
  PoolLayer,
  tensor2D,
  tensor3D,
} from "../mod.ts";

await setupBackend(CPU);

const input = tensor3D([
  [
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
  ],
  [
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
  ],
  [
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
  ],
]);

const output = tensor2D([
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]);

const net = new NeuralNetwork({
  silent: true,
  layers: [
    ConvLayer({
      activation: "tanh",
      kernelSize: [3, 3],
      padding: 0,
      strides: [1, 1],
    }),
    PoolLayer({ strides: [2, 2], mode: "max" }),
    DenseLayer({ size: [3], activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

net.train([
  {
    inputs: input,
    outputs: output,
  },
], 5000);

console.log(
  await net.predict(tensor3D([
    [
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 1, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
    ],
  ])),
);

console.log(
  await net.predict(tensor3D([
    [
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
    ],
  ])),
);

console.log(
  await net.predict(tensor3D([
    [
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
    ],
  ])),
);