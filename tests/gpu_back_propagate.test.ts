import { NeuralNetwork } from "../mod.ts";
import { GPUMatrix } from "../src/gpu/matrix.ts";
import { GPUNetwork } from "../src/gpu/network.ts";

const net = await new NeuralNetwork({
  hidden: [
    { size: 2, activation: "sigmoid" },
  ],
  cost: "crossentropy",
  output: { size: 2, activation: "sigmoid" },
}).setupBackend(true);

const network = (net.network as GPUNetwork);

await network.initialize("f32", 2, 3);
await network.feedForward(
  await GPUMatrix.from(
    network.backend,
    new Float32Array([
      1,
      2,
      3,
      4,
      5,
      6,
    ]),
    2,
    3,
  ),
);

const res = await network.backpropagate(
  await GPUMatrix.from(
    network.backend,
    new Float32Array([
      0,
      0,
      3,
      4,
      5,
      6,
    ]),
    2,
    3,
  ),
  0.1,
);

console.log(await res.data.get());
