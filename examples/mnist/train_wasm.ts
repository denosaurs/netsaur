import {
  Conv2DLayer,
  Cost,
  DenseLayer,
  FlattenLayer,
  Init,
  MaxPool2DLayer,
  Rank,
  ReluLayer,
  Sequential,
  setupBackend,
  SoftmaxLayer,
  Tensor,
  WASM,
} from "../../mod.ts";
import { loadDataset } from "./common.ts";

await setupBackend(WASM);

// training
const network = new Sequential({
  size: [1, 1, 28, 28],
  silent: true,
  layers: [
    Conv2DLayer({ kernelSize: [5, 5, 1, 6], padding: [2, 2] }),
    ReluLayer(),
    MaxPool2DLayer({ strides: [2, 2] }),
    Conv2DLayer({ kernelSize: [5, 5, 6, 16] }),
    ReluLayer(),
    MaxPool2DLayer({ strides: [2, 2] }),
    Conv2DLayer({ kernelSize: [5, 5, 16, 120] }),
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
const trainSet = loadDataset("train-images.idx", "train-labels.idx", 0, 5000);

const epochs = 1;
console.log("Training (" + epochs + " epochs)...");
const start = performance.now();
network.train(trainSet, epochs, 0.01);
console.log("Training complete!", performance.now() - start);

// predicting

const testSet = loadDataset("test-images.idx", "test-labels.idx", 0, 1000);
testSet.map((_, i) => (testSet[i].inputs.shape = [28, 28, 1]));

function argmax(mat: Tensor<Rank>) {
  let max = -Infinity;
  let index = -1;
  for (let i = 0; i < mat.data.length; i++) {
    if (mat.data[i] > max) {
      max = mat.data[i];
      index = i;
    }
  }
  return index;
}

const correct = testSet.filter(async (e) => {
  const prediction = argmax(await network.predict(e.inputs as Tensor<Rank>));
  const expected = argmax(e.outputs as Tensor<Rank>);
  return prediction === expected;
});

console.log(`${correct.length} / ${testSet.length} correct`);
console.log(
  `accuracy: ${((correct.length / testSet.length) * 100).toFixed(2)}%`
);
