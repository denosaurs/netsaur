import {
  Cost,
  CPU,
  DenseLayer,
  Sequential,
  setupBackend,
  SigmoidLayer,
  tensor2D,
} from "https://deno.land/x/netsaur@0.3.0/mod.ts";

import { parse } from "jsr:@std/csv@1.0.3/parse";

// Import helpers for metrics
import {
  // Metrics
  ClassificationReport,
  // Split the dataset
  useSplit,
} from "../../mod.ts";

// Define classes
const classes = ["Setosa", "Versicolor"];

// Read the training dataset
const _data = Deno.readTextFileSync("examples/metrics/binary_iris.csv");
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
  silent: true,

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

const res = await net.predict(tensor2D(test[0]));
const y1 = res.data.map(
  (x) => (x < 0.5 ? 0 : 1),
);
const cMatrix = new ClassificationReport(test[1], y1);
console.log("Confusion Matrix: ", cMatrix);
