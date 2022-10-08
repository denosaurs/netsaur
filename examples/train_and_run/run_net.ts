// import { CPU } from "../../backends/cpu.ts";
import { CPUBackend } from "../../src/cpu/backend.ts";

const net = CPUBackend.fromJSON(JSON.parse(Deno.readTextFileSync("./examples/train_and_run/network.json")));



console.log(await net.predict(new Float32Array([0, 0])));
console.log(await net.predict(new Float32Array([1, 0])));
console.log(await net.predict(new Float32Array([0, 1])));
console.log(await net.predict(new Float32Array([1, 1])));