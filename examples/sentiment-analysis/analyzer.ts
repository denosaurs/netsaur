import { CPU, setupBackend, tensor, Sequential } from "../../mod.ts";

import { type MatrixLike } from "jsr:@denosaurs/netsaur@0.4.0/utilities";

import { CategoricalEncoder } from "jsr:@denosaurs/netsaur@0.4.0/utilities/encoding";

import Mappings from "./mappings.json" with { type: "json" };
import Vocab from "./vocab.json" with { type: "json" };
import { TextVectorizer } from "../../packages/utilities/mod.ts";

const vocab = new Map();

for (const entry of Vocab.vocab) {
    vocab.set(entry[0], entry[1]);
}


const vectorizer = new TextVectorizer("indices");
vectorizer.mapper.mapping = vocab;
vectorizer.maxLength = Vocab.maxLength

const encoder = new CategoricalEncoder<string>();
const mappings = new Map();

for (const entry of Mappings) {
    mappings.set(entry[0], entry[1]);
}

encoder.mapping = mappings;

await setupBackend(CPU);

const net = Sequential.loadFile("examples/sentiment-analysis/sentiment.st");

const text = prompt("Text to analyze?") || "hello world";

const predYSoftmax = await net.predict(
    tensor(vectorizer.transform([text], "f32")),
);

CategoricalEncoder.fromSoftmax<"f32">(predYSoftmax as MatrixLike<"f32">);
const predY = encoder.untransform(predYSoftmax as MatrixLike<"f32">);

console.log(`The sentiment predicted is ${predY[0]}`);
