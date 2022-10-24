import {
  ConvLayer,
  DenseLayer,
  NeuralNetwork,
  PoolLayer,
  setupBackend,
  tensor2D,
  tensor1D,
} from "../mod.ts";
import { CPU } from "../backends/cpu/mod.ts";
import { ConvCPULayer } from "../backends/cpu/layers/conv.ts";
import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";
import { DenseCPULayer } from "../backends/cpu/layers/dense.ts";

await setupBackend(CPU);

const net = new NeuralNetwork({
  silent: true,
  layers: [
    ConvLayer({
      activation: "tanh",
      kernelSize: { x: 3, y: 3 },
      padding: 0,
      strides: 1,
    }),
    PoolLayer({ strides: 2 }),
    DenseLayer({ size: 10, activation: "sigmoid" }),
    DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  input: { x: 8, y: 8 },
});

const input_1 = await tensor2D([
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
]);

const input_2 = await tensor2D([
  [0, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1],
]);

const input_3 = await tensor2D([
  [1, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
]);

await net.train(
  [
    { inputs: input_1, outputs: await tensor1D([1]) },
    { inputs: input_2, outputs: await tensor2D([2]) },
    { inputs: input_3, outputs: await tensor2D([3]) },
  ],
  1,
  1,
);

console.log(await net.predict(input_1));
console.log(await net.predict(input_2));
console.log(await net.predict(input_3));
