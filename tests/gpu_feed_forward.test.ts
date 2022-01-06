import { WebGPUData } from "../deps.ts";
import { NeuralNetwork } from "../mod.ts";
import { GPUNetwork } from "../src/gpu/network.ts";

const time = Date.now()

const net = await new NeuralNetwork({
    hidden: [
        { size: 100, activation: "sigmoid" }
    ]
}).setupBackend(true);

for (let i = 0; i < 1000; i++) {
    const res = await (net.network as GPUNetwork).feedForward(
        await WebGPUData.from(
            (net.network as GPUNetwork).backend,
            new Float32Array(1000 * 100).fill(1)), 100, "f32"
    )
}

// console.log(await res.get())

console.log(`Time taken: ${Date.now() - time}ms`)