import { DenseLayer, NeuralNetwork } from "../../mod.ts";
import { CPU, CPUBackend } from "../../backends/cpu/mod.ts";

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    new DenseLayer({ size: 3, activation: "sigmoid" }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
}).setupBackend(CPU);

const time = performance.now();

await net.train(
  [
    { inputs: [0, 0, 1, 0, 0, 1, 1, 1], outputs: [0, 1, 1, 0] },
  ],
  5000,
  4,
  0.1,
);

console.log(`training time: ${performance.now() - time}ms`);

Deno.writeTextFileSync(
  "./examples/train_and_run/layer.json",
  JSON.stringify((net.backend as CPUBackend).layers[0].toJSON()),
);
Deno.writeTextFileSync(
  "./examples/train_and_run/output-layer.json",
  JSON.stringify((net.backend as CPUBackend).output.toJSON()),
);
