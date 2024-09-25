Deno.env.set("RUST_BACKTRACE", "1");
import {
  AdamOptimizer,
  Cost,
  CPU,
  Init,
  setupBackend,
  tensor,
  Sequential,
  DenseLayer,
  ReluLayer,
  SoftmaxLayer,
  EmbeddingLayer,
  FlattenLayer,
  NadamOptimizer,
} from "../../mod.ts";

import { CategoricalEncoder } from "jsr:@denosaurs/netsaur@0.4.0/utilities/encoding";

import { parse as parseCsv } from "jsr:@std/csv@1.0.3/parse";

import { format as duration } from "jsr:@std/fmt@1.0.2/duration";
import {
  TextVectorizer,
  useSplit,
  ClassificationReport,
  type MatrixLike,
  TextCleaner,
} from "../../packages/utilities/mod.ts";

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

const textCleaner = new TextCleaner({
  lowercase: true,
  normalizeWhiteSpaces: true,
  stripNewlines: true,
  removeStopWords: "english",
});
const cleanX = textCleaner.clean(text);

console.log("\nCSV Parsed");
console.timeLog("Time Elapsed");

const [[trainX, trainY], [testX, testY]] = useSplit(
  { shuffle: true, ratio: [7, 3] },
  cleanX,
  labels
);

console.log("Data Split");
console.timeLog("Time Elapsed");

const vectorizer = new TextVectorizer("indices");
const vecX = vectorizer.fit(trainX).transform(trainX, "f32");

const encoder = new CategoricalEncoder<string>();

const oneHotY = encoder.fit(trainY).transform(trainY, "f32");

Deno.writeTextFileSync(
  "examples/sentiment-analysis/mappings.json",
  JSON.stringify(Array.from(encoder.mapping.entries()))
);
Deno.writeTextFileSync(
  "examples/sentiment-analysis/vocab.json",
  JSON.stringify({vocab: Array.from(vectorizer.mapper.mapping.entries()), maxLength: vectorizer.maxLength})
);

console.log("\nCPU Backend Loading");
console.timeLog("Time Elapsed");

await setupBackend(CPU);

console.log("\nCPU Backend Loaded");
console.timeLog("Time Elapsed");

const net = new Sequential({
  size: [4, vecX.nCols],
  layers: [
    EmbeddingLayer({
      embeddingSize: 3,
      vocabSize: vectorizer.mapper.mapping.size,
    }),
    FlattenLayer({ size: [vecX.nCols * 3] }),
    DenseLayer({ size: [256], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [128], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [32], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [32], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [16], init: Init.Kaiming }),
    ReluLayer(),
    DenseLayer({ size: [encoder.mapping.size], init: Init.Kaiming }),
    SoftmaxLayer(),
  ],
  silent: false,
  optimizer: AdamOptimizer(),
  cost: Cost.CrossEntropy,
  patience: 10,
});

console.log("\nStarting");
console.timeLog("Time Elapsed");
const timeStart = performance.now();

net.train([{ inputs: tensor(vecX), outputs: tensor(oneHotY) }], 100, 2, 0.005);

console.log(
  `Training complete in ${duration(performance.now() - timeStart, {
    style: "narrow",
  })}.`
);

const predYSoftmax = await net.predict(
  tensor(vectorizer.transform(testX, "f32"))
);

CategoricalEncoder.fromSoftmax<"f32">(predYSoftmax as MatrixLike<"f32">);
const predY = encoder.untransform(predYSoftmax as MatrixLike<"f32">);

console.log(predY.map((x, i) => `${x}, ${testY[i]}`));

console.log(new ClassificationReport(testY, predY));

net.saveFile("examples/sentiment-analysis/sentiment.st");
