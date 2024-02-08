import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
import {
  Cost,
  CPU,
  DenseLayer,
  ReluLayer,
  Sequential,
  setupBackend,
  SigmoidLayer,
  tensor,
  tensor2D,
} from "../../mod.ts";

// Import helpers for metrics
import {
  ClassificationReport,
  TextVectorizer,
  // Split the dataset
  useSplit,
} from "https://deno.land/x/vectorizer@v0.3.7/mod.ts";

// Define classes
const ymap = ["spam", "ham"];

// Read the training dataset
const _data = Deno.readTextFileSync("examples/classification/spam.csv");
const data = parse(_data);

// Get the predictors (messages)
const x = data.map((msg) => msg[1]);

// Get the classes
const y = data.map((msg) => ymap.indexOf(msg[0]));

// Split the dataset for training and testing
const [train, test] = useSplit({ ratio: [7, 3], shuffle: true }, x, y) as [
  [typeof x, typeof y],
  [typeof x, typeof y],
];

// Vectorize the text messages
const vec = new TextVectorizer({
  mode: "tfidf",
  config: { skipWords: "english", standardize: { lowercase: true } },
}).fit(train[0]);

const x_vec = vec.transform(train[0], "f32");

// Setup the CPU backend for Netsaur
await setupBackend(CPU);

const net = new Sequential({
  size: [4, x_vec.nCols],
  layers: [
    // A dense layer with 256 neurons
    DenseLayer({ size: [256] }),
    // A relu activation layer
    ReluLayer(),
    // A dense layer with 8 neurons
    DenseLayer({ size: [32] }),
    // A relu activation layer
    ReluLayer(),
    // A dense layer with 8 neurons
    DenseLayer({ size: [8] }),
    // A relu activation layer
    ReluLayer(),
    // A dense layer with 1 neuron
    DenseLayer({ size: [1] }),
    // A sigmoid activation layer
    SigmoidLayer(),
  ],

  // We are using Log Loss for finding cost
  cost: Cost.BinCrossEntropy,
});

const inputs = tensor(x_vec.data, x_vec.shape);

const time = performance.now();
// Train the network
net.train(
  [
    {
      inputs: inputs,
      outputs: tensor2D(train[1].map((x) => [x])),
    },
  ],
  // Train for 20 epochs
  20,
  2,
  0.01,
);

console.log(`training time: ${performance.now() - time}ms`);

const x_vec_test = vec.transform(test[0]);

// Calculate metrics
const res = await net.predict(tensor(x_vec_test.data, x_vec_test.shape));
const y1 = res.data.map((i) => i < 0.5 ? 0 : 1);
const cMatrix = new ClassificationReport(test[1], y1);
console.log("Confusion Matrix: ", cMatrix);
