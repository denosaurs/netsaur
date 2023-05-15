import { Cost, CPU, DenseLayer, Sequential, setupBackend } from "../mod.ts";
await setupBackend(CPU);

Deno.test("save cpu", () => {
  const network = new Sequential({
    size: [4, 1],
    silent: true,
    layers: [
      DenseLayer({ size: [3] }),
    ],
    cost: Cost.MSE,
  });

  network.saveFile("./save_test_cpu.test.st");
});
