import {
  Activation,
  Cost,
  CPU,
  DenseLayer,
  Sequential,
  setupBackend,
} from "../mod.ts";

Deno.test("save cpu", async () => {
  await setupBackend(CPU);

  const network = new Sequential({
    size: [4, 1],
    silent: true,
    layers: [
      DenseLayer({ size: [3], activation: Activation.Linear }),
    ],
    cost: Cost.MSE,
  });

  network.save("./test.test.bin");
});
