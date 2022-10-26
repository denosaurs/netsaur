import {
  CPUTensor,
  LayerJSON,
  PoolLayerConfig,
  Rank,
  Shape,
  Shape2D,
} from "../../../core/types.ts";
import { average, iterate2D, maxIdx, to2D, to3D } from "../../../core/util.ts";
import { cpuZeroes3D } from "../../../mod.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Pooling layer.
 */
export class PoolCPULayer {
  outputSize!: Shape2D;
  strides: Shape2D;
  mode: "max" | "avg";
  indices: number[] = [];
  input!: CPUTensor<Rank.R3>;
  error!: CPUTensor<Rank.R3>;
  output!: CPUTensor<Rank.R3>;

  constructor(config: PoolLayerConfig) {
    this.strides = config.strides || [1, 1];
    this.mode = config.mode ?? "max";
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Shape[Rank]) {
    const size = to3D(inputSize);
    if (size[1] % this.strides[0] || size[2] % this.strides[1]) {
      throw new Error(
        `Cannot pool shape ${size} with stride ${this.strides}`,
      );
    }
    if (this.strides[0] == 1 || this.strides[1] == 1) {
      throw new Error(`Cannot pool with stride ${this.strides}`);
    }
    const w = size[1] / this.strides[0];
    const h = size[2] / this.strides[1];
    this.output = cpuZeroes3D([size[0], w, h]);
    this.outputSize = [h, w];
  }

  feedForward(inputTensor: CPUTensor<Rank>) {
    this.input = inputTensor.to3D();
    if (this.mode == "max") {
      iterate2D([this.output.y, this.output.z], (i: number, j: number) => {
        const pool: number[] = [];
        const indices: number[] = [];
        iterate2D(this.strides, (x: number, y: number) => {
          //TODO: batches
          const idx = (j * this.strides[0] + y) * this.input.y +
            i * this.strides[1] + x;
          pool.push(this.input.data[idx]);
          indices.push(idx);
        });
        const max = maxIdx(pool);
        this.indices[j * this.output.y + i] = indices[max];
        this.output.data[j * this.output.y + i] = pool[max];
      });
    } else if (this.mode == "avg") {
      iterate2D([this.output.y, this.output.z], (i: number, j: number) => {
        const pool: number[] = [];
        iterate2D(this.strides, (x: number, y: number) => {
          const idx = (j * this.strides[0] + y) * this.input.y +
            i * this.strides[1] + x;
          pool.push(this.input.data[idx]);
        });
        this.output.data[j * this.output.y + i] = average(pool);
      });
    }
    return this.output;
  }

  backPropagate(error: CPUTensor<Rank>, _rate: number) {
    this.error = error.to3D();
  }

  getError(): CPUTensor<Rank> {
    const error = cpuZeroes3D(this.input.shape);
    if (this.mode == "max") {
      for (let i = 0; i < this.error.data.length; i++) {
        error.data[this.indices[i]] = this.error.data[i];
      }
    } else if (this.mode == "avg") {
      iterate2D([this.output.y, this.output.z], (i: number, j: number) => {
        const meanError = this.error.data[j * this.error.x + i];
        iterate2D(this.strides, (x: number, y: number) => {
          const idx = (j * this.strides[0] + y) * error.x +
            i * this.strides[1] + x;
          error.data[idx] = meanError / this.strides[1] / this.strides[0];
        });
      });
    }
    return error;
  }

  toJSON() {
    return {
      outputSize: this.outputSize,
      type: "pool",
      strides: this.strides,
      mode: this.mode,
    };
  }

  static fromJSON(
    { outputSize, type, strides, mode }: LayerJSON,
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
    const layer = new PoolCPULayer({ strides: to2D(strides), mode });
    layer.outputSize = to2D(outputSize);
    return layer;
  }
}
