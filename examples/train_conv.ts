import {
  ConvLayer,
  DenseLayer,
  NeuralNetwork,
  PoolLayer,
  Rank,
  Tensor,
} from "../mod.ts";
import { ConvCPULayer } from "../backends/cpu/layers/conv.ts";
import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";
import { CPU, CPUMatrix } from "../backends/cpu/mod.ts";

const kernel = new Tensor<Rank.R2>([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
]);

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    new ConvLayer({
      activation: "linear",
      kernel: kernel.flatten(),
      kernelSize: { x: kernel.shape[1], y: kernel.shape[0] },
      padding: 2,
      strides: 2,
    }),
    new PoolLayer({ strides: 2 }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  input: 2,
}).setupBackend(CPU);

const buf = new Tensor<Rank.R2>([
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
]);

const input = new CPUMatrix(
  buf.flatten(),
  buf.shape[1],
  buf.shape[0],
);
const conv = net.getLayer(0) as ConvCPULayer;
const pool = net.getLayer(1) as PoolCPULayer;
net.initialize({ x: buf.shape[1], y: buf.shape[0] }, 1);
net.feedForward(input);

console.log(conv.padded.fmt());
console.log(conv.output.fmt());
console.log(pool.output.fmt());
