import { NeuralNetwork } from "../mod.ts";
import { CPUNetwork } from "../src/cpu/network.ts";

const time = Date.now()

const net = await new NeuralNetwork({
    hidden: [
        { size: 100, activation: "sigmoid" }
    ]
}).setupBackend(false);

for (let i = 0; i < 1000; i++) {
    const res = (net.network as CPUNetwork).feedForward(
        new Float32Array(1000* 100).fill(1), 100, "f32"
    )
}

// console.log(res)

console.log(`Time taken: ${Date.now() - time}ms`)