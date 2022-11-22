import { CPUBackend } from "../../backends/cpu/backend.ts";
import { CPU } from "../../backends/cpu/mod.ts";
import { setupBackend } from "../../core/mod.ts";
import { CPUTensor, Rank } from "../../core/types.ts";
import { loadDataset } from "./common.ts";

await setupBackend(CPU);

const network = CPUBackend.load("digit_model.json");

const testSet = loadDataset("test-images.idx", "test-labels.idx", 0, 1000);
testSet.map((_, i) => testSet[i].inputs.shape = [28, 28, 1]);

function argmax(mat: CPUTensor<Rank>) {
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

const correct = testSet.filter((e) => {
  const prediction = argmax(network.predict(e.inputs as CPUTensor<Rank>));
  const expected = argmax(e.outputs as CPUTensor<Rank>);
  return prediction === expected;
});

console.log(`${correct.length} / ${testSet.length} correct`);
console.log(
  `accuracy: ${((correct.length / testSet.length) * 100).toFixed(2)}%`,
);
