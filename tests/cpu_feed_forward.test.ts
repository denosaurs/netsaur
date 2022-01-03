import { NeuralNetwork } from "../mod.ts";
import { CPUNetwork } from "../src/cpu/network.ts";

const time = Date.now()

const net = await new NeuralNetwork({
    hidden: [
        { size: 3, activation: "sigmoid" }
    ]
}).setupBackend(false);

const res = (net.network as CPUNetwork).feedForward(
    new Float32Array([
        1, 2, // Batch 1
        3, 4, // Batch 2
        5, 6, // Batch 3
    ]), 3, "f32"
)

console.log(res)

console.log(`Time taken: ${Date.now() - time}ms`)