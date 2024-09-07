import { CPU, setupBackend, tensor } from "jsr:@denosaurs/netsaur@0.4.0";
import { Sequential } from "jsr:@denosaurs/netsaur@0.4.0/core";

import {
  useSplit,
  ClassificationReport,
  type MatrixLike,
} from "jsr:@denosaurs/netsaur@0.4.0/utilities";

import { CategoricalEncoder } from "jsr:@denosaurs/netsaur@0.4.0/utilities/encoding";
import {
  CountVectorizer,
  SplitTokenizer,
} from "jsr:@denosaurs/netsaur@0.4.0/utilities/text";

import Mappings from "./mappings.json" with {type: "json"}
import Vocab from "./vocab.json" with {type: "json"}
import Idf from "./idf.json" with {type: "json"}

import { parse as parseCsv } from "jsr:@std/csv@1.0.3/parse";


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

const [[_trainX, _trainY], [testX, testY]] = useSplit(
  { shuffle: true, ratio: [7, 3] },
  text,
  labels
);

console.log("Data Split");
console.timeLog("Time Elapsed");

const vocab = new Map();

for (const entry of Vocab) {
    vocab.set(entry[0], entry[1])
}

const tokenizer = new SplitTokenizer({
  skipWords: "english",
  vocabulary: vocab,
  standardize: { lowercase: true, stripNewlines: true },
});

const vectorizer = new CountVectorizer(tokenizer.vocabulary.size);

console.log("\nX vectorized");
console.timeLog("Time Elapsed");

const encoder = new CategoricalEncoder<string>();
const mappings = new Map();

for (const entry of Mappings) {
    mappings.set(entry[0], entry[1])
}

encoder.mapping = mappings;

console.log("\nCPU Backend Loading");
console.timeLog("Time Elapsed");

await setupBackend(CPU);

console.log("\nCPU Backend Loaded");
console.timeLog("Time Elapsed");

const net = Sequential.loadFile("examples/sentiment-analysis/sentiment.st")

const predYSoftmax = await net.predict(
  tensor(vectorizer.transform(tokenizer.transform(testX), "f32"))
);

CategoricalEncoder.fromSoftmax<"f32">(predYSoftmax as MatrixLike<"f32">);
const predY = encoder.untransform(predYSoftmax as MatrixLike<"f32">);

console.log(new ClassificationReport(testY, predY));

console.log(testY, predY)