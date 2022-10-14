import { Backend, NetworkJSON } from "../core/types.ts";
import { NeuralNetwork } from "../core/mod.ts";
import { ModelFormat } from "./types.ts";
import { JSONModel } from "./json.ts";

/**
 * Model Loader Class
 */
export class Model {
  static async load(
    path: string,
    helper?: {
      model: (data: NetworkJSON, silent: boolean) => Promise<Backend>;
    },
    format: ModelFormat = JSONModel,
    // deno-lint-ignore no-explicit-any
  ): Promise<any> {
    return await NeuralNetwork.fromJSON(
      await format.load(path),
      helper ? helper.model : undefined,
    );
  }

  static async save(
    path: string,
    net: NeuralNetwork,
    format: ModelFormat = JSONModel,
  ): Promise<void> {
    await format.save(path, net);
  }
}
