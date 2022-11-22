import {
  ConvLayer,
  DenseLayer,
  NeuralNetwork,
  PoolLayer,
  setupBackend,
} from "../../mod.ts";
import { loadDataset } from "./common.ts";
import { CPU } from "../../backends/cpu/mod.ts";
import { ReluLayer, SoftmaxLayer } from "../../layers/mod.ts";

await setupBackend(CPU);

const network = new NeuralNetwork({
  layers: [
    ConvLayer({ kernelSize: [5, 5, 1, 6], padding: 2 }),
    ReluLayer(),
    PoolLayer({ strides: [2, 2] }),
    ConvLayer({ kernelSize: [5, 5, 6, 16] }),
    ReluLayer(),
    PoolLayer({ strides: [2, 2] }),
    ConvLayer({ kernelSize: [5, 5, 16, 120] }),
    ReluLayer(),
    DenseLayer({ size: [84], init: "kaiming" }),
    ReluLayer(),
    DenseLayer({ size: [10], init: "kaiming" }),
    SoftmaxLayer(),
  ],
  cost: "crossentropy",
});

console.log("Loading training dataset...");
const trainSet = loadDataset("train-images.idx", "train-labels.idx", 0, 5000);

const epochs = 1;
console.log("Training (" + epochs + " epochs)...");
const start = performance.now();
await network.train(trainSet, epochs, 32, 0.01);
console.log("Training complete!", performance.now() - start);

network.save("digit_model.json");
