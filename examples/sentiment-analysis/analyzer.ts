import { CPU, setupBackend, tensor } from "jsr:@denosaurs/netsaur@0.4.0";
import { Sequential } from "jsr:@denosaurs/netsaur@0.4.0/core";

import type { MatrixLike } from "jsr:@denosaurs/netsaur@0.4.0/utilities";

import { CategoricalEncoder } from "jsr:@denosaurs/netsaur@0.4.0/utilities/encoding";
import {
    CountVectorizer,
    SplitTokenizer,
    TfIdfTransformer,
} from "jsr:@denosaurs/netsaur@0.4.0/utilities/text";

import Mappings from "./mappings.json" with { type: "json" };
import Vocab from "./vocab.json" with { type: "json" };
import Idf from "./tfidf.json" with { type: "json" };

const vocab = new Map();

for (const entry of Vocab) {
    vocab.set(entry[0], entry[1]);
}

const tokenizer = new SplitTokenizer({
    skipWords: "english",
    vocabulary: vocab,
    standardize: { lowercase: true, stripNewlines: true },
});

const vectorizer = new CountVectorizer(tokenizer.vocabulary.size);
const transformer = new TfIdfTransformer({ idf: Float64Array.from(Idf) });

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
    tensor(transformer.transform<"f32">(vectorizer.transform(tokenizer.transform([text]), "f32"))),
);

CategoricalEncoder.fromSoftmax<"f32">(predYSoftmax as MatrixLike<"f32">);
const predY = encoder.untransform(predYSoftmax as MatrixLike<"f32">);

console.log(`The sentiment predicted is ${predY[0]}`);
