import { Plugin } from "./types.ts";
import { Sequential } from "../../mod.ts";

/**
 * Load a plugin into a Sequential instance.
 */
export const loadPlugin = (
  instance: Sequential,
  loader: (instance: Sequential) => Plugin,
): Plugin => loader(instance);
