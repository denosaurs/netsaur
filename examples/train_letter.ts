import { CPU } from "../backends/cpu/mod.ts";
import { DataType, DataTypeArray } from "../deps.ts";
import {
  SigmoidLayer,
  DenseLayer,
  NeuralNetwork,
  setupBackend,
  tensor1D,
  tensor2D,
} from "../mod.ts";

// https://github.com/BrainJS/brain.js/blob/master/examples/typescript/which-letter-simple.ts
const character = (string: string): number[] =>
  string.trim().split("").map(integer);

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
    DenseLayer({ size: [10] }),
    SigmoidLayer(),
    DenseLayer({ size: [1] }),
    SigmoidLayer(),
  ],
  cost: "crossentropy",
});

await net.train(
  [
    {
      inputs: tensor2D([a, b, c]),
      outputs: tensor1D([
        "a".charCodeAt(0) / 255,
        "b".charCodeAt(0) / 255,
        "c".charCodeAt(0) / 255,
      ]),
    },
  ],
  10000,
);

console.log(toChar((await net.predict(tensor1D(a))).data));
console.log(toChar((await net.predict(tensor1D(b))).data));
console.log(toChar((await net.predict(tensor1D(c))).data));

function toChar<T extends DataType>(x: DataTypeArray<T>) {
  return String.fromCharCode(Math.round(x[0] * 255));
}
