import {
  Cost,
  WASM,
  DenseLayer,
  ReluLayer,
  RMSPropOptimizer,
  Sequential,
  setupBackend,
  SoftmaxLayer,
  type Tensor,
  tensor,
  tensor2D,
} from "../../packages/core/mod.ts";

import { parse } from "jsr:@std/csv@1.0.3/parse";

// Import helpers for metrics
import {
  // One-hot encoding of targets
  CategoricalEncoder,
  ClassificationReport,
  // Split the dataset
  useSplit,
} from "../../packages/utilities/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("examples/classification/iris.csv");
const data = parse(_data);

// Get the predictors (x) and targets (y)
const x = data.map((fl) => fl.slice(0, 4).map(Number));
const y_pre = data.map((fl) => fl[4]);

const encoder = new CategoricalEncoder();

const y = encoder.fit(y_pre).transform(y_pre, "f32");

// Split the dataset for training and testing
// @ts-ignore Matrices can be split
const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y);

// Setup the WebAssembly backend for Netsaur
await setupBackend(WASM);

console.log(train[1].head);

// Create a sequential neural network
const net = new Sequential({
  // Set number of minibatches to 4
  // Set size of output to 4
  size: [4, 4],

  // Disable logging during training
  silent: false,

  // Define each layer of the network
  layers: [
    // A dense layer with 8 neurons
    DenseLayer({ size: [8] }),
    // A ReLu activation layer
    ReluLayer(),
    // A dense layer with 8 neurons
    DenseLayer({ size: [8] }),
    // A ReLu activation layer
    ReluLayer(),
    // A dense layer with 3 neurons
    DenseLayer({ size: [3] }),
    // A Softmax activation layer
    SoftmaxLayer(),
  ],
  optimizer: RMSPropOptimizer(),
  // We are using CrossEntropy for finding cost
  cost: Cost.CrossEntropy,
});

const time = performance.now();

// Train the network
net.train(
  [
    {
      inputs: tensor2D(train[0]),
      outputs: tensor(train[1]),
    },
  ],
  // Train for 300 epochs
  400,
  1,
  0.02
);

console.log(`training time: ${performance.now() - time}ms`);

// Calculate metrics
const res = await net.predict(tensor2D(test[0]));
const y1 = encoder.untransform(res as Tensor<2>);
const y0 = encoder.untransform(test[1]);

const report = new ClassificationReport(y0, y1);
console.log(report);
