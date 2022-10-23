import {
  LayerJSON,
  PoolLayerConfig,
  Size,
  Size2D,
} from "../../../core/types.ts";
import { average, iterate2D, maxIdx, to2D } from "../../../core/util.ts";
import { CPUMatrix } from "../matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Pooling layer.
 */
export class PoolCPULayer {
  outputSize!: Size2D;
  strides: Size2D;
  mode: "max" | "avg";
  indices: number[] = [];
  input!: CPUMatrix;
  error!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: PoolLayerConfig) {
    this.strides = to2D(config.strides);
    this.mode = config.mode ?? "max";
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Size, _batches: number) {
    const w = (inputSize as Size2D).x / this.strides.x;
    const h = (inputSize as Size2D).y / this.strides.y;
    this.output = CPUMatrix.with(w, h);
    this.outputSize = { x: w, y: h };
  }

  feedForward(input: CPUMatrix) {
    if (this.mode == "max") {
      iterate2D(this.output, (i: number, j: number) => {
        const pool: number[] = [];
        const indices: number[] = [];
        iterate2D(this.strides, (x: number, y: number) => {
          const idx = (j * this.strides.y + y) * input.x +
            i * this.strides.x + x;
          pool.push(input.data[idx]);
          indices.push(idx);
        });
        const max = maxIdx(pool);
        this.indices[j * this.output.x + i] = indices[max];
        this.output.data[j * this.output.x + i] = pool[max];
      });
    } else if (this.mode == "avg") {
      iterate2D(this.output, (i: number, j: number) => {
        const pool: number[] = [];
        iterate2D(this.strides, (x: number, y: number) => {
          const idx = (j * this.strides.y + y) * input.x +
            i * this.strides.x + x;
          pool.push(input.data[idx]);
        });
        this.output.data[j * this.output.x + i] = average(pool);
      });
    }
    this.input = input;
    return this.output;
  }

  backPropagate(error: CPUMatrix, _rate: number) {
    this.error = error;
  }

  getError(): CPUMatrix {
    const error = CPUMatrix.with(this.input.x, this.input.y);
    if (this.mode == "max") {
      for (let i = 0; i < this.error.data.length; i++) {
        error.data[this.indices[i]] = this.error.data[i];
      }
    } else if (this.mode == "avg") {
      iterate2D(this.output, (i: number, j: number) => {
        const meanError = this.error.data[j * this.error.x + i];
        iterate2D(this.strides, (x: number, y: number) => {
          const idx = (j * this.strides.y + y) * error.x +
            i * this.strides.x + x;
          error.data[idx] = meanError / this.strides.x / this.strides.y;
        });
      });
    }
    return error;
  }

  toJSON(): LayerJSON {
    return {
      outputSize: this.outputSize,
      type: "pool",
      input: this.input.toJSON(),
      output: this.output.toJSON(),
      strides: this.strides,
      mode: this.mode,
    };
  }

  static fromJSON(
    { outputSize, input, type, output, strides, mode }: LayerJSON,
  ): PoolCPULayer {
    if (type !== "pool") {
      throw new Error(
        "Cannot cannot create a MaxPooling layer from a" +
          type.charAt(0).toUpperCase() + type.slice(1) +
          "Layer",
      );
    }
    if (strides === undefined) {
      throw new Error("Layer imported must be initialized");
    }
    const layer = new PoolCPULayer({ strides, mode });
    layer.input = new CPUMatrix(input.data, input.x, input.y);
    layer.outputSize = outputSize as Size2D;
    layer.output = new CPUMatrix(output.data, output.x, output.y);
    return layer;
  }
}
