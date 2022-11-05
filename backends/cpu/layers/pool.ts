import {
  CPUTensor,
  LayerJSON,
  PoolLayerConfig,
  Rank,
  Shape,
  Shape2D,
} from "../../../core/types.ts";
import { average, iterate1D, iterate2D, maxIdx } from "../../../core/util.ts";
import { cpuZeroes3D, reshape3D, toShape3D } from "../../../mod.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Pooling layer.
 */
export class PoolCPULayer {
  outputSize!: Shape2D;
  strides: Shape2D;
  mode: "max" | "avg";
  indices: number[][] = [];
  input!: CPUTensor<Rank.R3>;
  error!: CPUTensor<Rank.R3>;
  output!: CPUTensor<Rank.R3>;

  constructor(config: PoolLayerConfig) {
    this.strides = config.strides || [1, 1];
    this.mode = config.mode ?? "max";
  }

  reset(batches: number) {
    const [w, h] = this.outputSize
    this.output = cpuZeroes3D([w, h, batches]);
  }

  initialize(inputSize: Shape[Rank]) {
    const size = toShape3D(inputSize);
    if (size[0] % this.strides[0] || size[1] % this.strides[1]) {
      throw new Error(
        `Cannot pool shape ${size} with stride ${this.strides}`,
      );
    }
    if (this.strides[0] == 1 || this.strides[1] == 1) {
      throw new Error(`Cannot pool with stride ${this.strides}`);
    }
    const w = size[0] / this.strides[0];
    const h = size[1] / this.strides[1];
    this.outputSize = [w, h];
    this.reset(size[2])
  }

  feedForward(inputTensor: CPUTensor<Rank>) {
    this.input = reshape3D(inputTensor);
    if (this.mode == "max") {
      iterate1D(this.input.z, (z: number) => {
        this.indices[z] = [];
        iterate2D([this.output.x, this.output.y], (x: number, y: number) => {
          const pool: number[] = [];
          const indices: number[] = [];
          iterate2D(this.strides, (i: number, j: number) => {
            const w = x * this.strides[0] + i;
            const h = y * this.strides[1] + j;
            const idx = this.input.index(w, h, z);
            pool.push(this.input.data[idx]);
            indices.push(idx);
          });
          const idx = this.output.index(x, y, z);
          const max = maxIdx(pool);
          this.indices[z][idx] = indices[max];
          this.output.data[idx] = pool[max];
        });
      });
    } else if (this.mode == "avg") {
      iterate1D(this.input.z, (z: number) => {
        iterate2D([this.output.x, this.output.y], (x: number, y: number) => {
          const pool: number[] = [];
          iterate2D(this.strides, (i: number, j: number) => {
            const w = x * this.strides[0] + i;
            const h = y * this.strides[1] + j;
            const idx = this.input.index(w, h, z);
            pool.push(this.input.data[idx]);
          });
          this.output.data[this.output.index(x, y, z)] = average(pool);
        });
      });
    }
    return this.output;
  }

  backPropagate(errorTensor: CPUTensor<Rank>, _rate: number) {
    this.error = reshape3D(errorTensor);
    const error = cpuZeroes3D(this.input.shape);
    if (this.mode == "max") {
      iterate1D(this.input.z, (z: number) => {
        for (let i = 0; i < this.error.data.length; i++) {
          error.data[this.indices[z][i]] = this.error.data[i];
        }
      });
    } else if (this.mode == "avg") {
      iterate1D(this.input.z, (z: number) => {
        iterate2D([this.output.x, this.output.y], (x: number, y: number) => {
          const meanError = this.error.data[this.error.index(x, y, z)];
          iterate2D(this.strides, (i: number, j: number) => {
            const w = x * this.strides[0] + i;
            const h = y * this.strides[1] + j;
            const idx = this.error.index(w, h, z);
            error.data[idx] = meanError / this.strides[0] / this.strides[1];
          });
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
    const layer = new PoolCPULayer({ strides: strides as Shape2D, mode });
    layer.outputSize = outputSize as Shape2D;
    return layer;
  }
}
