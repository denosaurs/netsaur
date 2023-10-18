import type { NeuralNetwork, Rank, Tensor } from "../mod.ts";
import { Line } from "./types.ts";

/**
 * Visualizer for Neural Networks
 */
export class Visualizer {
  async graph<R extends Rank>(
    net: NeuralNetwork,
    inputs: Tensor<R>[],
    expectedResults: Tensor<R>[],
  ) {
    const expected: Line = {
      x: [],
      y: [],
      type: "scatter",
      name: "Expected",
      line: {
        color: "blue",
        width: 3,
      },
    };
    const results: Line = {
      x: [],
      y: [],
      type: "scatter",
      name: "Results",
      line: {
        color: "red",
        width: 3,
      },
    };

    for (let i = 0; i < inputs.length; i++) {
      expected.x.push(i + 1);
      results.x.push(i + 1);
      const output = (await net.predict(inputs[i])).data;
      expected.y.push(expectedResults[i].data[0]);
      results.y.push(output[0]);
    }

    return {
      [Symbol.for("Jupyter.display")]() {
        return {
          "application/vnd.plotly.v1+json": {
            data: [expected, results],
          },
        };
      },
    };
  }
}
