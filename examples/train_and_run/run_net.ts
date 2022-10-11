import { CPUModel } from "../../backends/cpu/mod.ts";
import { NeuralNetwork } from "../../mod.ts";

const net = await NeuralNetwork.fromJSON(
  JSON.parse(Deno.readTextFileSync("./examples/train_and_run/network.json")),
  CPUModel,
);

console.log(await net.predict(new Float32Array([0, 0])));
console.log(await net.predict(new Float32Array([1, 0])));
console.log(await net.predict(new Float32Array([0, 1])));
console.log(await net.predict(new Float32Array([1, 1])));
