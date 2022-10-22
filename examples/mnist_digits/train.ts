import { DenseLayer, NeuralNetwork, setupBackend } from "../../mod.ts";
import { Native } from "../../backends/native/mod.ts";
import { loadDataset } from "./common.ts";

await setupBackend(Native);
const network = new NeuralNetwork({
  input: 784,
  layers: [
    DenseLayer({ size: 28 * 2, activation: "sigmoid" }),
    DenseLayer({ size: 10, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

console.log("Loading training dataset...");
const trainSet = loadDataset("train-images.idx", "train-labels.idx");

const epochs = 5;
console.log("Training (" + epochs + " epochs)...");
const start = performance.now();
network.train(trainSet, epochs, 0.1);
console.log("Training complete!", performance.now() - start);

network.save("digit_model.bin");
