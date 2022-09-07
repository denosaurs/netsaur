import { assert } from "https://deno.land/std@0.154.0/testing/asserts.ts";
import { NeuralNetwork } from "../mod.ts";

let x = 0;

Deno.bench({
  name: "example test",
  fn(): void {
    x += 1;
    console.log(x);
    const net = new NeuralNetwork({
      silent: true,
      hidden: [
        { size: 6, activation: "sigmoid" },
        { size: 6, activation: "sigmoid" },
        // { size: 3, activation: "sigmoid" },
      ],
      cost: "crossentropy",
      output: { size: 1, activation: "sigmoid" },
      input: {
        type: "f32",
      },
    });

    net.train(
      [
        { inputs: [0, 0], outputs: [0] },
        { inputs: [1, 0], outputs: [1] },
        { inputs: [0, 1], outputs: [1] },
        { inputs: [1, 1], outputs: [0] },
      ],
      5000,
      1,
      0.1,
    );

    const a = net.predict(new Float32Array([0, 0]));
    const b = net.predict(new Float32Array([1, 0]));
    const c = net.predict(new Float32Array([0, 1]));
    const d = net.predict(new Float32Array([1, 1]));
    assert(a < 0.1, `${a}, ${b}, ${c}, ${d}, ${x}`);
    assert(b > 0.9, `${a}, ${b}, ${c}, ${d}, ${x}`);
    assert(c > 0.9, `${a}, ${b}, ${c}, ${d}, ${x}`);
    assert(d < 0.1, `${a}, ${b}, ${c}, ${d}, ${x}`);
  },
});
