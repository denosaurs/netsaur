import {
  LayerJSON,
  PoolLayerConfig,
  Size,
  Size2D,
} from "../../../core/types.ts";
import { to2D } from "../../../core/util.ts";
import { CPUMatrix } from "../matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * MaxPool layer.
 */
export class PoolCPULayer {
  outputSize!: Size2D;
  strides: Size2D;

  input!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: PoolLayerConfig) {
    this.strides = to2D(config.strides);
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
    for (let i = 0; i < this.output.x; i++) {
      for (let j = 0; j < this.output.y; j++) {
        const pool = [];
        for (let x = 0; x < this.strides.x; x++) {
          for (let y = 0; y < this.strides.y; y++) {
            const idx = (j * this.strides.y + y) * input.x +
              i * this.strides.x + x;
            pool.push(input.data[idx]);
          }
        }
        this.output.data[j * this.output.x + i] = Math.max(...pool);
      }
    }
    return this.output;
  }

  backPropagate(_error: CPUMatrix, _rate: number) {
  }

  toJSON(): LayerJSON {
    return {
      outputSize: this.outputSize,
      type: "pool",
      input: this.input.toJSON(),
      output: this.output.toJSON(),
      strides: this.strides,
    };
  }

  static fromJSON(
    { outputSize, input, output, strides }: LayerJSON,
  ): PoolCPULayer {
    const layer = new PoolCPULayer({ strides: strides });
    layer.input = new CPUMatrix(input.data, input.x, input.y);
    layer.outputSize = outputSize as Size2D;
    layer.output = new CPUMatrix(output.data, output.x, output.y);
    return layer;
  }
}
