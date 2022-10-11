import {
  LayerJSON,
  PoolLayerConfig,
  Size,
  Size2D,
} from "../../../core/types.ts";
import { CPUMatrix } from "../matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * MaxPool layer.
 */
export class PoolCPULayer {
  outputSize!: Size2D;
  stride: number;

  input!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: PoolLayerConfig) {
    this.stride = config.stride;
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Size, _batches: number) {
    const w = (inputSize as Size2D).x / this.stride;
    const h = (inputSize as Size2D).y / this.stride;
    this.output = CPUMatrix.with(w, h);
    this.outputSize = { x: w, y: h };
  }

  feedForward(input: CPUMatrix) {
    for (let i = 0; i < this.output.x; i++) {
      for (let j = 0; j < this.output.y; j++) {
        const pool = [];
        for (let x = 0; x < this.stride; x++) {
          for (let y = 0; y < this.stride; y++) {
            const idx = (j * this.stride + y) * input.x +
              i * this.stride + x;
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
      stride: this.stride,
    };
  }

  static fromJSON(
    { outputSize, input, output, stride }: LayerJSON,
  ): PoolCPULayer {
    const layer = new PoolCPULayer({ stride: stride as number });
    layer.input = new CPUMatrix(input.data, input.x, input.y);
    layer.outputSize = outputSize as Size2D;
    layer.output = new CPUMatrix(output.data, output.x, output.y);
    return layer;
  }
}
