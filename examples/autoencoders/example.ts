import { Matrix } from "../../packages/utilities/mod.ts";
import {
  AdamOptimizer,
  Cost,
  CPU,
  DenseLayer,
  ReluLayer,
  Sequential,
  setupBackend,
  type Shape2D,
  tensor,
} from "../../packages/core/mod.ts";

import { parse } from "jsr:@std/csv@1.0.3/parse";

const data = parse(
  Deno.readTextFileSync("examples/autoencoders/winequality-red.csv"),
);
data.shift();

const x_data = data.slice(0, 20).map((fl) => fl.slice(0, 11).map(Number));
const X = new Matrix(x_data, "f32");

await setupBackend(CPU);

const net = new Sequential({
  size: [4, X.nCols],
  silent: false,
  layers: [
    // Encoder
    DenseLayer({ size: [8] }),
    ReluLayer(),
    DenseLayer({ size: [4] }),
    ReluLayer(),
    DenseLayer({ size: [2] }),
    // Decoder
    DenseLayer({ size: [4] }),
    ReluLayer(),
    DenseLayer({ size: [8] }),
    ReluLayer(),
    DenseLayer({ size: [X.nCols] }),
  ],
  cost: Cost.MSE,
  patience: 50,
  optimizer: AdamOptimizer(),
  //  scheduler: OneCycle()
});

const input = tensor(X);

const timeStart = performance.now();
net.train([{ inputs: input, outputs: input }], 10000, 1, 0.01);
console.log(`Trained in ${performance.now() - timeStart}ms`);

function saveTable(name: string, data: Matrix<"f32">) {
  Deno.writeTextFileSync(`examples/autoencoders/${name}.html`, data.html);
}

saveTable("input", X);

console.log("Running Whole Net");
const output = await net.predict(input);

const output_mat = new Matrix<"f32">(output.data, output.shape as Shape2D);

saveTable("output", output_mat);

console.log("Running Encoder");
const encoded = await net.predict(input, [0, 5]);

const encoded_mat = new Matrix<"f32">(encoded.data, encoded.shape as Shape2D);

saveTable("encoded", encoded_mat);

console.log("Running Decoder");
const decoded = await net.predict(tensor(encoded_mat), [
  5,
  10,
]);

const decoded_mat = new Matrix<"f32">(decoded.data, decoded.shape as Shape2D);

saveTable("decoded", decoded_mat);
