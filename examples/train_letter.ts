import { DataType, DataTypeArray } from "../deps.ts";
import { NeuralNetwork } from "../mod.ts";

// https://github.com/BrainJS/brain.js/blob/master/examples/typescript/which-letter-simple.ts
const character = (string: string): Float32Array =>
  Float32Array.from(string.trim().split("").map(integer));

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
  silent: true,
  hidden: [
    { size: 3, activation: "sigmoid" },
  ],
  cost: "crossentropy",
  output: { size: 1, activation: "sigmoid" },
}).setupBackend(false);

net.train(
  [
    { inputs: a, outputs: ["a".charCodeAt(0) / 255] },
    { inputs: b, outputs: ["b".charCodeAt(0) / 255] },
    { inputs: c, outputs: ["c".charCodeAt(0) / 255] },
  ],
  5000,
  1,
  0.1,
);

console.log(toChar(net.predict(a)));
console.log(toChar(net.predict(b)));
console.log(toChar(net.predict(c)));

function toChar<T extends DataType>(x: DataTypeArray<T>) {
  return String.fromCharCode(Math.round(x[0] * 255));
}
