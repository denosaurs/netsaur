import {
AveragePool2DLayer,
  Conv2DLayer,
  Cost,
  CPU,
  MaxPool2DLayer,
  Rank,
  Sequential,
  setupBackend,
  Tensor,
  tensor4D,
} from "../../mod.ts";
import { decode } from "https://deno.land/x/pngs@0.1.1/mod.ts";
import { createCanvas } from "https://deno.land/x/canvas@v1.4.1/mod.ts";
import { Layer } from "../../src/core/api/layer.ts";

const canvas = createCanvas(600, 600);
const ctx = canvas.getContext("2d");
ctx.fillStyle = "white";
ctx.fillRect(0, 0, 600, 600);

const dim = 28;
const kernel = [
  [
    [
      [0, -1, 1],
      [0, -1, 1],
      [0, -1, 1],
    ],
  ],
];

//Credit: Hashrock (https://github.com/hashrock)
const img = decode(Deno.readFileSync("./examples/filters/deno.png")).image;
const buffer = new Float32Array(dim * dim);
for (let i = 0; i < dim * dim; i++) {
  buffer[i] = img[i * 4];
}

await setupBackend(CPU);

drawPixels(buffer, dim);

const conv = await feedForward([
  Conv2DLayer({
    kernel: tensor4D(kernel),
    kernelSize: [1, 1, 3, 3],
    padding: [1, 1],
  }),
]);

drawPixels(conv.data, conv.shape[2], 280);

const pool = await feedForward([
  Conv2DLayer({
    kernel: tensor4D(kernel),
    kernelSize: [1, 1, 3, 3],
    padding: [1, 1],
  }),
  MaxPool2DLayer({ strides: [2, 2] }),
]);
drawPixels(pool.data, pool.shape[2], 0, 280, 2);

const pool2 = await feedForward([
  Conv2DLayer({
    kernel: tensor4D(kernel),
    kernelSize: [1, 1, 3, 3],
    padding: [1, 1],
  }),
  AveragePool2DLayer({ strides: [2, 2] }),
]);
drawPixels(pool2.data, pool2.shape[2], 280, 280, 2);


async function feedForward(layers: Layer[]) {
  const net = new Sequential({
    size: [1, 1, dim, dim],
    silent: true,
    layers,
    cost: Cost.MSE,
  });

  const data = new Tensor(buffer, [1, 1, dim, dim]);
  return (await net.predict(data)) as Tensor<Rank.R4>;
}

function drawPixels(
  buffer: Float32Array,
  dim: number,
  offsetX = 0,
  offsetY = 0,
  scale = 1,
) {
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      const pixel = buffer[j * dim + i];
      ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
      ctx.fillRect(
        i * 10 * scale + offsetX,
        j * 10 * scale + offsetY,
        10 * scale,
        10 * scale,
      );
    }
  }
}

await Deno.writeFile("./examples/filters/output.png", canvas.toBuffer());
