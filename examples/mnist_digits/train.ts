import { ConvLayer, DenseLayer, NeuralNetwork, PoolLayer, setupBackend } from "../../mod.ts";
import { loadDataset } from "./common.ts";
import { CPU } from "../../backends/cpu/mod.ts";
import { Softmax } from "../../layers/mod.ts";

await setupBackend(CPU);

const network = new NeuralNetwork({
  layers: [
    ConvLayer({ kernelSize: [5, 5, 1, 6], padding: 2, activation: "relu" }),
    PoolLayer({ strides: [2, 2] }),
    ConvLayer({ kernelSize: [5, 5, 6, 16], activation: "relu" }),
    PoolLayer({ strides: [2, 2] }),
    ConvLayer({ kernelSize: [5, 5, 16, 120], activation: "relu" }),
    DenseLayer({ size: [84], activation: "relu" }),
    DenseLayer({ size: [10], activation: "linear" }),
    Softmax()
  ],
  cost: "crossentropy",
});

console.log("Loading training dataset...");
const trainSet = loadDataset("train-images.idx", "train-labels.idx", 0, 10000);

const epochs = 1;
console.log("Training (" + epochs + " epochs)...");
const start = performance.now();
await network.train(trainSet, epochs, 32, 0.01);
console.log("Training complete!", performance.now() - start);

network.save("digit_model.json");
