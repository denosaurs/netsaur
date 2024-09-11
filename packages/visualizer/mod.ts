import type { NeuralNetwork, Rank, Tensor } from "../../mod.ts";
import type { Line } from "./types.ts";

/**
 * Visualizer for Neural Networks in Jupyter Notebook
 */
export class Visualizer {
  #title: string;

  constructor(title: string) {
    this.#title = title;
  }

  /**
   * Graph the results of a Neural Network
   */
  async graph<R extends Rank>(
    net: NeuralNetwork,
    inputs: Tensor<R>[],
    expectedResults: Tensor<R>[],
  ): Promise<
    {
      [x: symbol]: () => {
        "application/vnd.plotly.v1+json": {
          data: Line[];
          layout: { title: string };
        };
      };
    }
  > {
    const expected: Line = {
      x: [],
      y: [],
      type: "scatter",
      mode: "lines+markers",
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
      mode: "lines+markers",
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
    const title = this.#title;

    return {
      [Symbol.for("Jupyter.display")]() {
        return {
          "application/vnd.plotly.v1+json": {
            data: [expected, results],
            layout: {
              title,
            },
          },
        };
      },
    };
  }

  /**
   * Graph the loss of a Neural Network during training
   */
  graphLoss(loss: number[]): {
    [x: symbol]: () => {
      "application/vnd.plotly.v1+json": {
        data: Line[];
        layout: { title: string };
      };
    };
  } {
    const lossLine: Line = {
      x: [],
      y: [],
      type: "scatter",
      mode: "lines+markers",
      name: "Loss",
      line: {
        color: "blue",
        width: 3,
      },
    };

    for (let i = 0; i < loss.length; i++) {
      lossLine.x.push(i + 1);
      lossLine.y.push(loss[i]);
    }

    const title = this.#title;

    return {
      [Symbol.for("Jupyter.display")]() {
        return {
          "application/vnd.plotly.v1+json": {
            data: [lossLine],
            layout: {
              title,
            },
          },
        };
      },
    };
  }
}
