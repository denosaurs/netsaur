import { DenseLayer, NeuralNetwork } from "../../mod.ts";
import { CPU } from "../../backends/cpu/mod.ts";

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    DenseLayer.fromJSON(
      JSON.parse(Deno.readTextFileSync("./examples/train_and_run/layer.json")),
    ),
    DenseLayer.fromJSON(
      JSON.parse(
        Deno.readTextFileSync("./examples/train_and_run/output-layer.json"),
      ),
    ),
  ],
  cost: "crossentropy",
}).setupBackend(CPU);

console.log(await net.predict(new Float32Array([0, 0])));
console.log(await net.predict(new Float32Array([1, 0])));
console.log(await net.predict(new Float32Array([0, 1])));
console.log(await net.predict(new Float32Array([1, 1])));

