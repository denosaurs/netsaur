import { DenseLayer, NeuralNetwork } from "../../mod.ts";
import { Native } from "../../backends/native.ts";
import { loadDataset } from "./common.ts";

const network = await new NeuralNetwork({
  input: 784,
  layers: [
    new DenseLayer({ size: 28 * 2, activation: "sigmoid" }),
    new DenseLayer({ size: 10, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
}).setupBackend(Native);

console.log("Loading training dataset...");
const trainSet = loadDataset("train-images.idx", "train-labels.idx");

const epochs = 5;
console.log("Training (" + epochs + " epochs)...");
const start = performance.now();
network.train(trainSet, epochs, 0.1);
console.log("Training complete!", performance.now() - start);

network.save("digit_model.bin");
