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
const _b = character(
  "######." +
    "#.....#" +
    "#.....#" +
    "######." +
    "#.....#" +
    "#.....#" +
    "######.",
);
const _c = character(
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
    { size: 2, activation: "relu" },
  ],
  cost: "crossentropy",
  output: { size: 2, activation: "relu" },
}).setupBackend(false);

net.train(
  {
    inputs: a,
    outputs: new Uint32Array(["a".charCodeAt(0)]),
  },
  1,
  4,
  0.1,
);
console.log((net.network as CPUNetwork).output);

// TODO: predict

