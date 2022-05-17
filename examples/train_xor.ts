import { NeuralNetwork } from "../mod.ts";

const time = Date.now();

const net = await new NeuralNetwork({
  hidden: [
    { size: 3, activation: "sigmoid" },
  ],
  cost: "crossentropy",
  output: { size: 1, activation: "sigmoid" },
  input: {
    type: "f32",
  },
}).setupBackend(false);

await net.train(
  [
    { inputs: [0, 0], outputs: [0] },
    { inputs: [1, 0], outputs: [1] },
    { inputs: [0, 1], outputs: [1] },
    { inputs: [1, 1], outputs: [0] },
  ],
  5000,
  1,
  0.3,
);

console.log(net.predict(new Float32Array([0, 0])));
console.log(net.predict(new Float32Array([1, 0])));
console.log(net.predict(new Float32Array([0, 1])));
console.log(net.predict(new Float32Array([1, 1])));

console.log(Date.now() - time);
