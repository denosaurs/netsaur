import type { NeuralNetwork } from "../mod.ts";

// deno-lint-ignore no-unused-vars
class Image {
  #data: string;
  constructor(data: string) {
    this.#data = data;
  }
  [Symbol.for("Jupyter.display")]() {
    return {
      "image/png": this.#data,
    };
  }
}

export class Visualizer {
  show(_net: NeuralNetwork) {
    // const b64 = (Deno as any)[(Deno as any).internal].core.ops.op_base64_encode(
    //   data,
    // );
    // return new Image(b64);
    throw new Error("Not implemented");
  }
}
