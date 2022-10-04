import { NativeBackend } from "../../src/native/backend.ts";
import { DataType, Matrix } from "../../src/native/matrix.ts";
import { loadDataset } from "./common.ts";

const network = NativeBackend.load("digit_model.bin");

const testSet = loadDataset("test-images.idx", "test-labels.idx");

function argmax<T extends DataType>(mat: Matrix<T>) {
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
  const prediction = argmax(network.predict(e.inputs));
  const expected = argmax(e.outputs);
  return prediction === expected;
});

console.log(`${correct.length} / ${testSet.length} correct`);
console.log(`accuracy: ${((correct.length / testSet.length) * 100).toFixed(2)}%`);