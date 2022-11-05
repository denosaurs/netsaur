import { ConvLayer, DenseLayer, NeuralNetwork, PoolLayer, setupBackend } from "../../mod.ts";
import { loadDataset } from "./common.ts";
import { CPU } from "../../backends/cpu/mod.ts";

await setupBackend(CPU);

const network = new NeuralNetwork({
  layers: [
    ConvLayer({
      activation: "tanh",
      kernelSize: [3, 3],
      padding: 0,
      strides: [1, 1],
    }),
    PoolLayer({ strides: [2, 2], mode: "max" }),
    DenseLayer({ size: [10], activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

console.log("Loading training dataset...");
const trainSet = loadDataset("train-images.idx", "train-labels.idx");

const epochs = 5;
console.log("Training (" + epochs + " epochs)...");
const start = performance.now();
await network.train(trainSet, epochs, 0.1);
console.log("Training complete!", performance.now() - start);

network.save("digit_model.json");
