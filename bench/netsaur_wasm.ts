import {
  Cost,
  DenseLayer,
  Sequential,
  setupBackend,
  SigmoidLayer,
  tensor1D,
  tensor2D,
  WASM,
} from "../mod.ts";

await setupBackend(WASM);

const net = new Sequential({
  size: [4, 2],
  silent: true,
  layers: [
    DenseLayer({ size: [3] }),
    SigmoidLayer(),
    DenseLayer({ size: [1] }),
    SigmoidLayer(),
  ],
  cost: Cost.MSE,
});

net.train(
  [
    {
      inputs: tensor2D([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ]),
      outputs: tensor2D([[0], [1], [1], [0]]),
    },
  ],
  10000,
);

console.log((await net.predict(tensor1D([0, 0]))).data);
console.log((await net.predict(tensor1D([1, 0]))).data);
console.log((await net.predict(tensor1D([0, 1]))).data);
console.log((await net.predict(tensor1D([1, 1]))).data);
