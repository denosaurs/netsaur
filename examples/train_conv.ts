import { ConvLayer, DenseLayer, NeuralNetwork, PoolLayer } from "../mod.ts";
import { ConvCPULayer } from "../src/cpu/layers/conv.ts";
import { PoolCPULayer } from "../src/cpu/layers/pool.ts";
import { CPUMatrix } from "../src/cpu/matrix.ts";
import { CPUBackend } from "../src/cpu/backend.ts";
import { CPU } from "../src/cpu/mod.ts";

const kernel = new Float32Array([
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
]);

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    new ConvLayer({
      activation: "sigmoid",
      kernel: kernel,
      kernelSize: { x: 3, y: 3 },
      padding: 2,
      stride: 2,
    }),
    new PoolLayer({ stride: 2 }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  input: 2,
}).setupBackend(CPU);

const buf = new Float32Array([
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
]);
const input = new CPUMatrix(buf, 5, 5);
const network = net.backend as CPUBackend;
const conv = network.layers[0] as ConvCPULayer;
const pool = network.layers[1] as PoolCPULayer;
network.initialize({ x: 5, y: 5 }, 1);
network.layers[0].feedForward(input);
network.layers[1].feedForward(conv.output);
console.log(conv.padded.fmt());
console.log(conv.output.fmt());
console.log(pool.output.fmt());
