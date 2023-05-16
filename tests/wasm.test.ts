import { Cost, DenseLayer, Sequential, setupBackend, SigmoidLayer, tensor1D, tensor2D, WASM } from "../mod.ts";

await setupBackend(WASM);

Deno.test("save wasm", () => {
  const network = new Sequential({
    size: [4, 1],
    silent: true,
    layers: [
      DenseLayer({ size: [3] }),
    ],
    cost: Cost.MSE,
  });

  network.saveFile("./save_test_wasm.test.st");
});


Deno.test("cost(wasm): hinge", async () => {
  const net = new Sequential({
    size: [4, 2],
    silent: true,
    layers: [
      DenseLayer({ size: [3] }),
      SigmoidLayer(),
      DenseLayer({ size: [1] }),
      SigmoidLayer(),
    ],
    cost: Cost.Hinge,
  });

  net.train([{
    inputs: tensor2D([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]),
    outputs: tensor2D([[0], [1], [1], [0]]),
  }], 50000);

  const out1 = (await net.predict(tensor1D([0, 0]))).data;
  console.log(`0 xor 0 = ${out1[0]} (should be close to 0)`);

  const out2 = (await net.predict(tensor1D([1, 0]))).data;
  console.log(`1 xor 0 = ${out2[0]} (should be close to 1)`);

  const out3 = (await net.predict(tensor1D([0, 1]))).data;
  console.log(`0 xor 1 = ${out3[0]} (should be close to 1)`);

  const out4 = (await net.predict(tensor1D([1, 1]))).data;
  console.log(`1 xor 1 = ${out4[0]} (should be close to 0)`);
});

Deno.test("cost(wasm): mse", async () => {
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

  net.train([{
    inputs: tensor2D([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]),
    outputs: tensor2D([[0], [1], [1], [0]]),
  }], 5000);

  const out1 = (await net.predict(tensor1D([0, 0]))).data;
  console.log(`0 xor 0 = ${out1[0]} (should be close to 0)`);

  const out2 = (await net.predict(tensor1D([1, 0]))).data;
  console.log(`1 xor 0 = ${out2[0]} (should be close to 1)`);

  const out3 = (await net.predict(tensor1D([0, 1]))).data;
  console.log(`0 xor 1 = ${out3[0]} (should be close to 1)`);

  const out4 = (await net.predict(tensor1D([1, 1]))).data;
  console.log(`1 xor 1 = ${out4[0]} (should be close to 0)`);
});
