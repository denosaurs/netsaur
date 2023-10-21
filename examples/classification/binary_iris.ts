import {
  Cost,
  CPU,
  DenseLayer,
  Sequential,
  setupBackend,
  SigmoidLayer,
  tensor1D,
  tensor2D,
} from "../../mod.ts";

import { parse } from "https://deno.land/std@0.204.0/csv/parse.ts";

// Import helpers for metrics
import {
  accuracyScore,
  // Metrics
  ConfusionMatrix,
  precisionScore,
  sensitivityScore,
  specificityScore,
  // Split the dataset
  useSplit,
} from "https://deno.land/x/vectorizer@v0.2.1/mod.ts";

// Define classes
const classes = ["Setosa", "Versicolor"];

// Read the training dataset
const _data = Deno.readTextFileSync("examples/classification/binary_iris.csv");
const data = parse(_data);

// Get the predictors (x) and targets (y)
const x = data.map((fl) => fl.slice(0, 4).map(Number));
const y = data.map((fl) => classes.indexOf(fl[4]));

// Split the dataset for training and testing
const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y) as [
  [typeof x, typeof y],
  [typeof x, typeof y],
];

// Setup the CPU backend for Netsaur
await setupBackend(CPU);

// Create a sequential neural network
const net = new Sequential({
  // Set number of minibatches to 4
  // Set size of output to 4
  size: [4, 4],

  // Disable logging during training
  silent: false,

  // Define each layer of the network
  layers: [
    // A dense layer with 4 neurons
    DenseLayer({ size: [4] }),
    // A sigmoid activation layer
    SigmoidLayer(),
    // A dense layer with 1 neuron
    DenseLayer({ size: [1] }),
    // Another sigmoid layer
    SigmoidLayer(),
  ],
  // We are using MSE for finding cost
  cost: Cost.MSE,
});

const time = performance.now();

// Train the network
net.train(
  [
    {
      inputs: tensor2D(train[0]),
      outputs: tensor2D(train[1].map((x) => [x])),
    },
  ],
  // Train for 10000 epochs
  10000,
);

console.log(`training time: ${performance.now() - time}ms`);

// Calculate metrics
let [tp, fn, fp, tn] = [0, 0, 0, 0];
for (let i = 0; i < test[0].length; i += 1) {
  const res = (await net.predict(tensor1D(test[0][i]))).data[0] < 0.5 ? 0 : 1;
  if (res === 1 && test[1][i] == 1) tp += 1;
  if (res === 0 && test[1][i] == 1) fn += 1;
  if (res === 1 && test[1][i] == 0) fp += 1;
  if (res === 0 && test[1][i] == 0) tn += 1;
}

const cMatrix = new ConfusionMatrix([tp, fn, fp, tn]);
console.log("Confusion Matrix: ", cMatrix);
console.log("Accuracy: ", `${accuracyScore(cMatrix) * 100}%`);
console.log("Precision: ", `${precisionScore(cMatrix) * 100}%`);
console.log("Sensitivity / Recall: ", `${sensitivityScore(cMatrix) * 100}%`);
console.log("Specificity: ", `${specificityScore(cMatrix) * 100}%`);
