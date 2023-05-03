import {
  Activation,
  Cost,
  CPU,
  DenseLayer,
  Sequential,
  setupBackend,
  tensor2D,
} from "../mod.ts";

await setupBackend(CPU);

const net = new Sequential({
  size: [4, 2],
  silent: true,
  layers: [
    DenseLayer({ size: [3], activation: Activation.Sigmoid }),
    DenseLayer({ size: [1], activation: Activation.Sigmoid }),
  ],
  cost: Cost.MSE,
});

const time = performance.now();

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
)

console.log(`training time: ${performance.now() - time}ms`);
console.log((await net.predict(tensor2D([[0, 0]]))).data);
console.log((await net.predict(tensor2D([[1, 0]]))).data);
console.log((await net.predict(tensor2D([[0, 1]]))).data);
console.log((await net.predict(tensor2D([[1, 1]]))).data);
