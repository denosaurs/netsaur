import {
  Activation,
  Cost,
  CPU,
  DenseLayer,
  NeuralNetwork,
  setupBackend,
  tensor2D,
} from "../mod.ts";

await setupBackend(CPU);

Deno.bench(
  { name: "xor 5000 epochs", permissions: "inherit" },
  async () => {
    const net = new NeuralNetwork({
      size: [4, 2],
      silent: true,
      layers: [
        DenseLayer({ size: [3], activation: Activation.Sigmoid }),
        DenseLayer({ size: [1], activation: Activation.Sigmoid }),
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
      5000,
    );

    console.log((await net.predict(tensor2D([[0, 0]]))).data);
    console.log((await net.predict(tensor2D([[1, 0]]))).data);
    console.log((await net.predict(tensor2D([[0, 1]]))).data);
    console.log((await net.predict(tensor2D([[1, 1]]))).data);
  },
);

Deno.bench(
  { name: "xor 10000 epochs", permissions: "inherit" },
  async () => {
    const net = new NeuralNetwork({
      size: [4, 2],
      silent: true,
      layers: [
        DenseLayer({ size: [3], activation: Activation.Sigmoid }),
        DenseLayer({ size: [1], activation: Activation.Sigmoid }),
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

    console.log((await net.predict(tensor2D([[0, 0]]))).data);
    console.log((await net.predict(tensor2D([[1, 0]]))).data);
    console.log((await net.predict(tensor2D([[0, 1]]))).data);
    console.log((await net.predict(tensor2D([[1, 1]]))).data);
  },
);
