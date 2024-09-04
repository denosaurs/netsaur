import {
  AdamOptimizer,
  Cost,
  CPU,
  DenseLayer,
  OneCycle,
  Sequential,
  setupBackend,
  tensor1D,
  tensor2D,
} from "../../mod.ts";

import { parse } from "https://deno.land/std@0.214.0/csv/parse.ts";

// Import helpers for splitting dataset
import { useSplit } from "https://deno.land/x/vectorizer@v0.2.1/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("examples/multiple-linear/student.csv");
const data = parse(_data);

// Get the independent variables (x) and map text to numbers
const x = data.map((fl) =>
  [fl[0], fl[1], fl[2] === "Yes" ? 1 : 0, fl[3], fl[4]].map(Number)
);
// Get dependent variables (y)
const y = data.map((fl) => Number(fl[5]));

// Split the data for training and testing
// Cast to original types because useSplit returns a weird type
const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y) as [
  [typeof x, typeof y],
  [typeof x, typeof y],
];

// Setup the CPU backend for Netsaur
await setupBackend(CPU);

console.log(train)

// Create a sequential neural network
const net = new Sequential({
  // Set number of minibatches to 4
  // Set size of output to 5
  size: [4, 5],

  // Disable logging during training
  silent: true,

  // Set up a simple linear model
  layers: [
    // A dense layer with 8 neurons
    DenseLayer({ size: [8] }),
    // A dense layer with 1 neuron
    DenseLayer({ size: [1] }),
  ],
  // We are using Adam as the optimizer
  optimizer: AdamOptimizer(),
  // The one cycle scheduler cycles between max_rate and
  // the learning rate supplied during training.
  scheduler: OneCycle({ max_rate: 0.05, step_size: 100 }),
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
  // Train for 1000 epochs
  1000,
  1,
  0.02,
);

console.log(`training time: ${performance.now() - time}ms`);

// Compute RMSE
let err = 0;
const y_test = await net.predict(tensor2D(test[0]));
for (const i in test[0]) {
  err += (test[1][i] - y_test.data[i]) ** 2;
  console.log(`\nOutput: ${y_test.data[i]}\nExpected: ${test[1][i]}`);
}
console.log("RMSE:", Math.sqrt(err / test[0].length));
