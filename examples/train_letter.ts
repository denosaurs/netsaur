import { CPU } from "../backends/cpu/mod.ts";
import { DataType, DataTypeArray } from "../deps.ts";
import {
  DenseLayer,
  NeuralNetwork,
  setupBackend,
  tensor1D,
  tensor2D,
} from "../mod.ts";

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
await setupBackend(CPU);
const net = new NeuralNetwork({
  silent: true,
  layers: [
    DenseLayer({ size: 10, activation: "sigmoid" }),
    DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

net.train(
  [
    {
      inputs: await tensor2D([a, b, c]),
      outputs: await tensor1D([
        "a".charCodeAt(0) / 255,
        "b".charCodeAt(0) / 255,
        "c".charCodeAt(0) / 255,
      ]),
    },
  ],
  5000,
  1,
  0.1,
);

console.log(toChar(await net.predict(a)));
console.log(toChar(await net.predict(b)));
console.log(toChar(await net.predict(c)));

function toChar<T extends DataType>(x: DataTypeArray<T>) {
  return String.fromCharCode(Math.round(x[0] * 255));
}
