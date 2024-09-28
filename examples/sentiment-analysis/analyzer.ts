import { CPU, setupBackend, tensor, Sequential, Tensor } from "../../mod.ts";

import Mappings from "./mappings.json" with { type: "json" };
import Vocab from "./vocab.json" with { type: "json" };
import { TextVectorizer, CategoricalEncoder } from "../../packages/utilities/mod.ts";

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

encoder.mapper.mapping = mappings;

await setupBackend(CPU);

const net = Sequential.loadFile("examples/sentiment-analysis/sentiment.st");

const text = prompt("Text to analyze?") || "hello world";

const predYSoftmax = await net.predict(
    tensor(vectorizer.transform([text], "f32")),
);

const predY = encoder.untransform(predYSoftmax as Tensor<2>);

console.log(`The sentiment predicted is ${predY[0]}`);
