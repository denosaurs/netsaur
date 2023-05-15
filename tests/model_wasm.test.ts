import { Cost, DenseLayer, Sequential, setupBackend, WASM } from "../mod.ts";

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
