import { DenseCPULayer } from "../backends/cpu/layers/dense.ts";
import { CPUMatrix } from "../backends/cpu/matrix.ts";
import { iterate1D } from "../core/util.ts";
import { DataTypeArray } from "../deps.ts";
import { CrossEntropy, DenseLayer, tensor1D, tensor2D } from "../mod.ts";

const layer = DenseLayer({ size: 3, activation: "sigmoid" }) as DenseCPULayer;
const output = DenseLayer({ size: 1, activation: "sigmoid" }) as DenseCPULayer;

const datasets = [{
  inputs: await tensor2D([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ]),
  outputs: await tensor1D([0, 1, 1, 0]),
}];


const batches = datasets[0].inputs.y;
const inputSize = datasets[0].inputs.x;
const epochs = 5000;
const rate = 0.1;
layer.initialize(inputSize, batches);
output.initialize(layer.outputSize, batches);
const time = performance.now();

// train
iterate1D(epochs, (_e: number) => {
  for (const dataset of datasets) {
    let input = dataset.inputs;
    input = layer.feedForward(input);
    output.feedForward(input);
    const out = dataset.outputs;
    let error = CPUMatrix.with(output.output.x, output.output.y);
    for (const i in output.output.data) {
      error.data[i] = CrossEntropy(
        out[i],
        output.output.data[i],
        true,
      );
    }
    output.backPropagate(error, rate);
    const weights = output.weights;
    error = CPUMatrix.dot(error, CPUMatrix.transpose(weights));
    layer.backPropagate(error, rate);
  }
});
console.log(`training time: ${performance.now() - time}ms`);

// predict

function predict(data: DataTypeArray) {
  const input = new CPUMatrix(data, data.length, 1);
  layer.reset(1);
  output.reset(1);
  return output.feedForward(layer.feedForward(input)).data;
}

console.log(predict(new Float32Array([0, 0])));
console.log(predict(new Float32Array([1, 0])));
console.log(predict(new Float32Array([0, 1])));
console.log(predict(new Float32Array([1, 1])));
