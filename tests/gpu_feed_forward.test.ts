import { WebGPUData } from "../deps.ts";
import { NeuralNetwork } from "../mod.ts";
import { GPUNetwork } from "../src/gpu/network.ts";

const time = Date.now()

const net = await new NeuralNetwork({
    hidden: [
        { size: 3, activation: "sigmoid" }
    ]
}).setupBackend(true);

const res = await (net.network as GPUNetwork).feedForward(
    await WebGPUData.from(
        (net.network as GPUNetwork).backend,
        new Float32Array([
        1, 2, // Batch 1
        3, 4, // Batch 2
        5, 6, // Batch 3
    ])), 3, "f32"
)

console.log(await res.get())

console.log(`Time taken: ${Date.now() - time}ms`)