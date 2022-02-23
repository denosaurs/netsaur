import { NeuralNetwork } from "../mod.ts";

const net = await new NeuralNetwork({
    hidden: [
        { size: 3, activation: "relu" }
    ],
    cost: "crossentropy"
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
}, 10, 4);