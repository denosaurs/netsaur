import { DenseLayer, NeuralNetwork } from "../mod.ts";

// cpu

const cpuTime = Date.now();

const cpuNet = await new NeuralNetwork({
  hidden: [
    new DenseLayer({ size: 3, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  output: new DenseLayer({ size: 1, activation: "sigmoid" }),
  input: {
    type: "f32",
  },
}).setupBackend("cpu");

cpuNet.train(
  [
    { inputs: [0, 0, 1, 0, 0, 1, 1, 1], outputs: [0, 1, 1, 0] },
  ],
  5000,
  4,
  0.1,
);

console.log(await cpuNet.predict(new Float32Array([0, 0])));
console.log(await cpuNet.predict(new Float32Array([1, 0])));
console.log(await cpuNet.predict(new Float32Array([0, 1])));
console.log(await cpuNet.predict(new Float32Array([1, 1])));
const cpuResult = Date.now() - cpuTime;
console.log(cpuResult);

// gpu
const time = Date.now();

const net = await new NeuralNetwork({
  hidden: [
    new DenseLayer({ size: 3, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  output: new DenseLayer({ size: 1, activation: "sigmoid" }),
  input: {
    type: "f32",
  },
}).setupBackend("gpu");

await net.train(
  [
    { inputs: [0, 0, 1, 0, 0, 1, 1, 1], outputs: [0, 1, 1, 0] },
  ],
  5000,
  4,
  0.1,
);

console.log(await net.predict(new Float32Array([0, 0])));
console.log(await net.predict(new Float32Array([1, 0])));
console.log(await net.predict(new Float32Array([0, 1])));
console.log(await net.predict(new Float32Array([1, 1])));

const gpuResult = Date.now() - time;
console.log(gpuResult);

console.log(`${Math.round((cpuResult / gpuResult) * 100)}% speed increase`);
