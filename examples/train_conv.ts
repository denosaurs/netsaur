import {
  ConvLayer,
  DenseLayer,
  NeuralNetwork,
  PoolLayer,
  tensor2D,
} from "../mod.ts";
import { ConvCPULayer } from "../backends/cpu/layers/conv.ts";
import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";
import { CPU } from "../backends/cpu/mod.ts";

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    new ConvLayer({
      activation: "linear",
      kernel: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
      kernelSize: { x: 3, y: 3 },
      padding: 2,
      strides: 2,
    }),
    new PoolLayer({ strides: 2 }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  input: 2,
}).setupBackend(CPU);

const input = await tensor2D([
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
]);

const conv = net.getLayer(0) as ConvCPULayer;
const pool = net.getLayer(1) as PoolCPULayer;
net.initialize(input, 1);
net.feedForward(input);

console.log(conv.padded.fmt());
console.log(conv.output.fmt());
console.log(pool.output.fmt());
