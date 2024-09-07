import { AdamOptimizer, Cost, CPU, Init, setupBackend, tensor } from "jsr:@denosaurs/netsaur@0.4.0";
import { Sequential } from "jsr:@denosaurs/netsaur@0.4.0/core";
import { NadamOptimizer } from "jsr:@denosaurs/netsaur@0.4.0/core/optimizers";
import {
  DenseLayer,
  ReluLayer,
  SoftmaxLayer,
} from "jsr:@denosaurs/netsaur@0.4.0/core/layers";

import {
  useSplit,
  ClassificationReport,
  MatrixLike,
} from "jsr:@denosaurs/netsaur@0.4.0/utilities";

import { CategoricalEncoder } from "jsr:@denosaurs/netsaur@0.4.0/utilities/encoding";
import {
  CountVectorizer,
  TfIdfTransformer,
  SplitTokenizer,
} from "jsr:@denosaurs/netsaur@0.4.0/utilities/text";

import { parse as parseCsv } from "jsr:@std/csv@1.0.3/parse";

import { format as duration } from "jsr:@std/fmt@1.0.2/duration";

console.time("Time Elapsed");

console.log("\nImports loaded.");

const file = Deno.readTextFileSync(
  "examples/sentiment-analysis/text_emotion.csv"
);

console.log("\nData file loaded.");
console.timeLog("Time Elapsed");

const data = parseCsv(file, { skipFirstRow: true }) as {
  sentiment: string;
  content: string;
}[];

const text = data.map((x) => x.content);
const labels = data.map((x) => x.sentiment);

console.log("\nCSV Parsed");
console.timeLog("Time Elapsed");

const [[trainX, trainY], [testX, testY]] = useSplit(
  { shuffle: true, ratio: [7, 3] },
  text,
  labels
);

console.log("Data Split");
console.timeLog("Time Elapsed");

const tokenizer = new SplitTokenizer({
  skipWords: "english",
  standardize: { lowercase: true, stripNewlines: true },
});

const tokens = tokenizer.fit(trainX).transform(trainX);

console.log("\nX tokenized");
console.timeLog("Time Elapsed");

const vectorizer = new CountVectorizer(tokenizer.vocabulary.size);

const vecX = vectorizer.transform(tokens, "f32");

tokens.splice(0, tokens.length);

console.log("\nX vectorized");
console.timeLog("Time Elapsed");

const transformer = new TfIdfTransformer();

const tfidfX = transformer.fit(vecX).transform<"f32">(vecX);

console.log("\nX Transformed", tfidfX.shape);
console.timeLog("Time Elapsed");

const encoder = new CategoricalEncoder<string>(); 

const oneHotY = encoder.fit(trainY).transform(trainY, "f32");

Deno.writeTextFileSync("examples/sentiment-analysis/mappings.json", JSON.stringify(Array.from(encoder.mapping.entries())))
Deno.writeTextFileSync("examples/sentiment-analysis/vocab.json", JSON.stringify(Array.from(tokenizer.vocabulary.entries())))
Deno.writeTextFileSync("examples/sentiment-analysis/tfidf.json", JSON.stringify(transformer.idf))

console.log("\nCPU Backend Loading");
console.timeLog("Time Elapsed");

await setupBackend(CPU);

console.log("\nCPU Backend Loaded");
console.timeLog("Time Elapsed");

const net = new Sequential({
  size: [4, vecX.nCols],
  layers: [
    DenseLayer({ size: [256], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [32], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [16], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [16], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [16], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [encoder.mapping.size], init: Init.Kaiming }),
    SoftmaxLayer(),
  ],
  silent: false,
  optimizer: AdamOptimizer(),
  cost: Cost.CrossEntropy,
  patience: 10
});

console.log("\nStarting");
console.timeLog("Time Elapsed");
const timeStart = performance.now();

net.train([{ inputs: tensor(tfidfX), outputs: tensor(oneHotY) }], 100, 2, 0.002);

console.log(
  `Training complete in ${duration(performance.now() - timeStart, {
    style: "narrow",
  })}.`
);

const predYSoftmax = await net.predict(
  tensor(transformer.transform<"f32">(vectorizer.transform(tokenizer.transform(testX), "f32")))
);

CategoricalEncoder.fromSoftmax<"f32">(predYSoftmax as MatrixLike<"f32">);
const predY = encoder.untransform(predYSoftmax as MatrixLike<"f32">);

console.log(new ClassificationReport(testY, predY));

net.saveFile("examples/sentiment-analysis/sentiment.st")