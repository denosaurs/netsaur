import { NeuralNetwork } from "../core/mod.ts";
import { NetworkJSON } from "../core/types.ts";
import type { ModelFormat } from "./types.ts";
import { staticImplements } from "./util.ts";
import proto from "./proto/keras_proto.js";

// deno-lint-ignore no-unused-vars
function fromKeraslayerType(type: string): string {
  switch (type) {
    case "Conv2D":
      return "conv";
    default:
      throw new Error("Unknown Layer type: " + type);
  }
}

@staticImplements<ModelFormat>()
export class KerasModel {
  static async load(
    path: string,
  ): Promise<NetworkJSON> {
    const _config: Partial<NetworkJSON> = {};
    if (!path.endsWith(".bin")) path = path + ".bin";
    const model = proto.Model.decode(await Deno.readFile(path));
    const _model_config = JSON.parse(model.modelConfig);
    throw new Error("Keras not yet implemented");
  }

  // deno-lint-ignore require-await
  static async save(_path: string, _net: NeuralNetwork) {
    throw new Error("Keras not yet implemented");
  }
}
