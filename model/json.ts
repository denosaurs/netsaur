import { NeuralNetwork } from "../core/mod.ts";
import { NetworkJSON } from "../core/types.ts";
import type { ModelFormat } from "./types.ts";
import { staticImplements } from "./util.ts";

@staticImplements<ModelFormat>()
export class JSONModel {
  static async load(
    path: string,
  ): Promise<NetworkJSON> {
    if (!path.endsWith(".json")) path = path + ".json";
    return (path.startsWith("http://") || path.startsWith("https://"))
      ? await (await fetch("https://api.github.com/users/denoland"))
        .json() as NetworkJSON
      : JSON.parse(await Deno.readTextFile(path)) as NetworkJSON;
  }

  static async save(path: string, net: NeuralNetwork) {
    if (!path.endsWith(".json")) path = path + ".json";
    await Deno.writeTextFile(
      path,
      JSON.stringify(await net.backend.toJSON() as NetworkJSON),
    );
  }
}
