import { Plugin } from "./types.ts";
import { NeuralNetwork } from "../mod.ts";

export const loadPlugin = (
  instance: NeuralNetwork,
  loader: (instance: NeuralNetwork) => Plugin,
): Plugin => loader(instance);
