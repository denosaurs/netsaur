import { NeuralNetwork } from "../mod.ts";

const net = await new NeuralNetwork({
    hidden: [
        { size: 2, activation: "sigmoid" }
    ],
    cost: "crossentropy",
    output: { size: 1, activation: "sigmoid" }
}).setupBackend(false)

net.train({
    inputs: new Float32Array([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ]),
    outputs: new Float32Array([
        0,
        1,
        1,
        0
    ])
}, 10000, 4, 1);

console.log(net.network.getOutput())