import { CPU, Rank, Sequential, setupBackend, Tensor } from "../../mod.ts";
import { loadDataset } from "./common.ts";

await setupBackend(CPU);

const network = Sequential.loadFile("examples/mnist/mnist.test.st");

const testSet = loadDataset("test-images.idx", "test-labels.idx", 0, 1000);
testSet.map((_, i) => (testSet[i].inputs.shape = [1, 28, 28]));

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

let correct = 0;
for (const test of testSet) {
  const prediction = argmax(await network.predict(test.inputs as Tensor<Rank>));
  const expected = argmax(test.outputs as Tensor<Rank>);
  if (expected === prediction) {
    correct += 1;
  }
}

console.log(`${correct} / ${testSet.length} correct`);
console.log(
  `accuracy: ${((correct / testSet.length) * 100).toFixed(2)}%`,
);
