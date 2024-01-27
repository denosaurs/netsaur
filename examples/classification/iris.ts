Deno.env.set("RUST_BACKTRACE", "1");

import {
  AdamOptimizer,
  Cost,
  CPU,
  DenseLayer,
  OneCycle,
  ReluLayer,
  Sequential,
  setupBackend,
  SoftmaxLayer,
  tensor,
  tensor1D,
  tensor2D,
} from "../../mod.ts";

import { parse } from "https://deno.land/std@0.204.0/csv/parse.ts";

// Import helpers for metrics
import {
  ClassificationReport,
  // Split the dataset
  useSplit,
  // One-hot encoding of targets
  CategoricalEncoder,
} from "https://deno.land/x/vectorizer@v0.3.5/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("examples/classification/iris.csv");
const data = parse(_data);

// Get the predictors (x) and targets (y)
const x = data.map((fl) => fl.slice(0, 4).map(Number));
const y_pre = data.map((fl) => fl[4]);

const encoder = new CategoricalEncoder()

const y = encoder.fit(y_pre).transform(y_pre, "f32")

// Split the dataset for training and testing
// @ts-ignore Matrices can be split
const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y) as [
  [typeof x, typeof y],
  [typeof x, typeof y]
];

// Setup the CPU backend for Netsaur
await setupBackend(CPU);

console.log(train[1])

// Create a sequential neural network
const net = new Sequential({
  // Set number of minibatches to 4
  // Set size of output to 4
  size: [4, 4],

  // Disable logging during training
  silent: false,

  // Define each layer of the network
  layers: [
    // A dense layer with 16 neurons
    DenseLayer({ size: [16] }),
    // A ReLu activation layer
    ReluLayer(),
    // A dense layer with 3 neurons
    DenseLayer({ size: [3] }),
    // A Softmax activation layer
    SoftmaxLayer(),
  ],
  optimizer: AdamOptimizer(),
  // We are using CrossEntropy for finding cost
  cost: Cost.CrossEntropy,
  scheduler: OneCycle({ max_rate: 0.05, step_size: 50 }),
});

const time = performance.now();

// Train the network
net.train(
  [
    {
      inputs: tensor2D(train[0]),
      outputs: tensor(train[1].data as Float32Array, train[1].shape),
    },
  ],
  // Train for 300 epochs
  400,
  1,
  0.02,
);

console.log(`training time: ${performance.now() - time}ms`);

// Calculate metrics
const res = await Promise.all(
  test[0].map((input) => net.predict(tensor1D(input))),
);
const y1 = res.map((x) => encoder.getOg(x.data.indexOf(Math.max(...x.data))));
const y0 = encoder.untransform(test[1]);

console.log(y1.map((x, i) => [y0[i], x]));
const cMatrix = new ClassificationReport(y0, y1);
console.log(cMatrix);
console.log(
  "Total Accuracy: ",
  y1.filter((x, i) => x === y0[i]).length / y1.length,
);
