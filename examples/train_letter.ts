import { NeuralNetwork } from "../mod.ts";
import { CPUNetwork } from "../src/cpu/network.ts";

// https://github.com/BrainJS/brain.js/blob/master/examples/typescript/which-letter-simple.ts
const character = (string: string): Uint32Array => Uint32Array.from(string.trim().split("").map(integer));

const integer = (character: string): number => character === "#" ? 1 : 0;


const a = character(
  ".#####." +
  "#.....#" +
  "#.....#" +
  "#######" +
  "#.....#" +
  "#.....#" +
  "#.....#",
);
const b = character(
  "######." +
  "#.....#" +
  "#.....#" +
  "######." +
  "#.....#" +
  "#.....#" +
  "######.",
);
const c = character(
  "#######" +
  "#......" +
  "#......" +
  "#......" +
  "#......" +
  "#......" +
  "#######",
);
const net = await new NeuralNetwork({
  hidden: [
    { size: 10, activation: "sigmoid" },
  ],
  cost: "crossentropy",
  output: { size: 1, activation: "sigmoid" },
}).setupBackend(false);

net.train([
  { inputs: a, outputs: ["a".charCodeAt(0)] },
  { inputs: b, outputs: ["b".charCodeAt(1)] },
  { inputs: c, outputs: ["c".charCodeAt(2)] },
], 1000, 1, 0.1);

console.log(net.predict(a));
console.log(net.predict(b));
console.log(net.predict(c));

