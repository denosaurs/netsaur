import type { Plugin } from "./types.ts";
import type { NeuralNetwork } from "../../../../mod.ts";

/**
 * Load a plugin into a NeuralNetwork instance.
 */
export const loadPlugin = (
  instance: NeuralNetwork,
  loader: (instance: NeuralNetwork) => Plugin,
): Plugin => loader(instance);
