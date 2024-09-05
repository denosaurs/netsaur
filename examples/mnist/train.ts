import {
  Conv2DLayer,
  Cost,
  CPU,
  DenseLayer,
  FlattenLayer,
  Init,
  MaxPool2DLayer,
  ReluLayer,
  Sequential,
  setupBackend,
  SoftmaxLayer,
} from "../../packages/core/mod.ts";

import { loadDataset } from "./common.ts";

await setupBackend(CPU);

// training
const network = new Sequential({
  size: [32, 1, 28, 28],
  layers: [
    Conv2DLayer({ kernelSize: [6, 1, 5, 5], padding: [2, 2] }),
    ReluLayer(),
    MaxPool2DLayer({ strides: [2, 2] }),
    Conv2DLayer({ kernelSize: [16, 6, 5, 5] }),
    ReluLayer(),
    MaxPool2DLayer({ strides: [2, 2] }),
    Conv2DLayer({ kernelSize: [120, 16, 5, 5] }),
    ReluLayer(),
    FlattenLayer({ size: [120] }),
    DenseLayer({ size: [84], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [10], init: Init.Kaiming }),
    SoftmaxLayer(),
  ],
  cost: Cost.CrossEntropy,
});

console.log("Loading training dataset...");
const trainSet = loadDataset(
  "train-images.idx",
  "train-labels.idx",
  0,
  5000,
  32,
);

const epochs = 1;
console.log("Training (" + epochs + " epochs)...");
const start = performance.now();
network.train(trainSet, epochs, 1, 0.005);
console.log("Training complete!", performance.now() - start);

network.saveFile("examples/mnist/mnist.test.st");
