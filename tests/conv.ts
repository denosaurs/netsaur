import { ConvLayer, DenseLayer, NeuralNetwork } from "../mod.ts";
import { ConvCPULayer } from "../src/cpu/layers/conv.ts";
import { CPUMatrix } from "../src/cpu/matrix.ts";
import { CPUBackend } from "../src/cpu/backend.ts";
import { CPU } from "../backends/cpu.ts";
import { PoolCPULayer } from "../src/cpu/layers/pool.ts";
import { PoolLayer } from "../src/mod.ts";

import { decode } from "https://deno.land/x/pngs@0.1.1/mod.ts";
import { DataTypeArray } from "../deps.ts";


import { Canvas } from "https://deno.land/x/neko@1.1.3/canvas/mod.ts";

const canvas = new Canvas({
  title: "Netsaur Convolutions",
  width: 600,
  height: 600,
  fps: 60,
});

const ctx = canvas.getContext("2d");
ctx.fillStyle = "white";
ctx.fillRect(0, 0, 600, 600);

const dim = 28;

//Credit: Hashrock (https://github.com/hashrock)
const img = decode(Deno.readFileSync("./tests/deno.png")).image;
const buf = new Float32Array(dim * dim) as DataTypeArray<"f32">;

for (let i = 0; i < dim * dim; i++) {
  buf[i] = img[i * 4];
}

const kernel = new Float32Array([
  -1,
  1,
  0,
  -1,
  1,
  0,
  -1,
  1,
  0,
]) as DataTypeArray<"f32">;

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    new ConvLayer({
      activation: "sigmoid",
      kernel: kernel,
      kernelSize: { x: 3, y: 3 },
      padding: 1,
      stride: 1,
    }),
    new PoolLayer({ stride: 2 }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  input: 2,
}).setupBackend(CPU);

const input = new CPUMatrix(buf, dim, dim);
const network = net.backend as CPUBackend;
const conv = network.layers[0] as ConvCPULayer;
const pool = network.layers[1] as PoolCPULayer;
network.initialize({ x: dim, y: dim }, 1);
network.layers[0].feedForward(input);
network.layers[1].feedForward(conv.output);
const cv = conv.output;
const out = pool.output;

for (let i = 0; i < dim; i++) {
  for (let j = 0; j < dim; j++) {
    const pixel = buf[j * dim + i];
    ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
    ctx.fillRect(i * 10, j * 10, 10, 10);
  }
}

for (let i = 0; i < cv.x; i++) {
  for (let j = 0; j < cv.y; j++) {
    const pixel = Math.round(Math.max(Math.min(cv.data[j * cv.x + i], 255), 0));
    ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
    ctx.fillRect(i * 10 + dim * 10, j * 10, 10, 10);
  }
}

for (let i = 0; i < out.x; i++) {
  for (let j = 0; j < out.y; j++) {
    const pixel = Math.round(
      Math.max(Math.min(out.data[j * out.x + i], 255), 0),
    );
    ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
    ctx.fillRect(i * 20 + dim * 10, j * 20 + dim * 10, 20, 20);
  }
}
