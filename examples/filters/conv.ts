import { ConvLayer, NeuralNetwork, Tensor } from "../../mod.ts";
import { ConvCPULayer } from "../../backends/cpu/layers/conv.ts";
import { PoolCPULayer } from "../../backends/cpu/layers/pool.ts";
import { PoolLayer } from "../../layers/mod.ts";
import { CPU } from "../../backends/cpu/mod.ts";

import { decode } from "https://deno.land/x/pngs@0.1.1/mod.ts";
import { DataTypeArray } from "../../deps.ts";

import { Canvas } from "https://deno.land/x/neko@1.1.3/canvas/mod.ts";
import { setupBackend } from "../../core/mod.ts";
// import { createCanvas } from "https://deno.land/x/canvas@v1.4.1/mod.ts";

// const canvas = createCanvas(600, 600);
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
const img = decode(Deno.readFileSync("./examples/filters/deno.png")).image;
const buf = new Float32Array(dim * dim) as DataTypeArray<"f32">;

for (let i = 0; i < dim * dim; i++) {
  buf[i] = img[i * 4];
}

const kernel = [
  [-1, 1, 0],
  [-1, 1, 0],
  [-1, 1, 0],
].flat();

setupBackend(CPU)

const net = new NeuralNetwork({
  silent: true,
  layers: [
    ConvLayer({
      activation: "linear",
      kernel: new Float32Array(kernel),
      kernelSize: [3, 3, 1, 1],
      padding: 1,
      strides: [1, 1],
      unbiased: true,
    }),
    PoolLayer({ strides: [2, 2], mode: "max" }),
  ],
  cost: "crossentropy",
});

const input = new Tensor(buf, [dim, dim, 1, 1]);

const conv = net.getLayer(0) as ConvCPULayer;
const pool = net.getLayer(1) as PoolCPULayer;

net.initialize([dim, dim, 1, 1], 1);
net.feedForward(input);

for (let i = 0; i < dim; i++) {
  for (let j = 0; j < dim; j++) {
    const pixel = buf[j * dim + i];
    ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
    ctx.fillRect(i * 10, j * 10, 10, 10);
  }
}

for (let i = 0; i < conv.output.x; i++) {
  for (let j = 0; j < conv.output.y; j++) {
    const pixel = Math.round(
      Math.max(Math.min(conv.output.data[j * conv.output.x + i], 255), 0),
    );
    ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
    ctx.fillRect(i * 10 + dim * 10, j * 10, 10, 10);
  }
}

for (let i = 0; i < pool.output.x; i++) {
  for (let j = 0; j < pool.output.y; j++) {
    const pixel = Math.round(
      Math.max(Math.min(pool.output.data[j * pool.output.x + i], 255), 0),
    );
    ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
    ctx.fillRect(i * 20 + dim * 10, j * 20 + dim * 10, 20, 20);
  }
}
// await Deno.writeFile("./examples/filters/output.png", canvas.toBuffer());
