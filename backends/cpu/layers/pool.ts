import {
  CPUTensor,
  LayerJSON,
  PoolLayerConfig,
  Rank,
  Shape,
  Shape2D,
  Shape3D,
} from "../../../core/types.ts";
import {
  average,
  iterate1D,
  iterate2D,
  iterate4D,
  maxIdx,
} from "../../../core/util.ts";
import { cpuZeroes4D, reshape4D, toShape4D } from "../../../mod.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Pooling layer.
 */
export class PoolCPULayer {
  outputSize!: Shape3D;
  strides: Shape2D;
  mode: "max" | "avg";
  indices!: CPUTensor<Rank.R4>;
  input!: CPUTensor<Rank.R4>;
  error!: CPUTensor<Rank.R4>;
  output!: CPUTensor<Rank.R4>;

  constructor(config: PoolLayerConfig) {
    this.strides = config.strides || [2, 2];
    this.mode = config.mode ?? "max";
  }

  reset(batches: number) {
    const [w, h, f] = this.outputSize;
    this.indices = cpuZeroes4D([w, h, f, batches]);
    this.output = cpuZeroes4D([w, h, f, batches]);
  }

  initialize(inputSize: Shape[Rank]) {
    const size = toShape4D(inputSize);
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
    this.outputSize = [w, h, size[2]];
    this.reset(size[3]);
  }

  feedForward(inputTensor: CPUTensor<Rank>) {
    this.input = reshape4D(inputTensor);
    if (this.mode == "max") {
      iterate4D(this.output, (x, y, z, w) => {
        const pool: number[] = [];
        const indices: number[] = [];
        iterate2D(this.strides, (i: number, j: number) => {
          const W = x * this.strides[0] + i;
          const H = y * this.strides[1] + j;
          const idx = this.input.index(W, H, z, w);
          pool.push(this.input.data[idx]);
          indices.push(idx);
        });
        const idx = this.output.index(x, y, z, w);
        const max = maxIdx(pool);
        this.indices.data[idx] = indices[max];
        this.output.data[idx] = pool[max];
      });
    } else if (this.mode == "avg") {
      iterate4D(this.output, (x, y, z, w) => {
        const pool: number[] = [];
        iterate2D(this.strides, (i, j) => {
          const W = x * this.strides[0] + i;
          const H = y * this.strides[1] + j;
          const idx = this.input.index(W, H, z, w);
          pool.push(this.input.data[idx]);
        });
        this.output.data[this.output.index(x, y, z, w)] = average(pool);
      });
    }
    return this.output;
  }

  backPropagate(errorTensor: CPUTensor<Rank>, _rate: number) {
    this.error = reshape4D(errorTensor);
    const error = cpuZeroes4D(this.input.shape);
    if (this.mode == "max") {
      iterate1D(this.error.data.length, (i) => {
        error.data[this.indices.data[i]] = this.error.data[i];
      });
    } else if (this.mode == "avg") {
      iterate4D(this.output, (x, y, z, w) => {
        const meanError = this.error.data[this.error.index(x, y, z, w)];
        iterate2D(this.strides, (i, j) => {
          const W = x * this.strides[0] + i;
          const H = y * this.strides[1] + j;
          const idx = this.error.index(W, H, z, w);
          error.data[idx] = meanError / this.strides[0] / this.strides[1];
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
    layer.outputSize = outputSize as Shape3D;
    return layer;
  }
}
