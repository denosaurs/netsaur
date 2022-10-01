import { ConvLayerConfig, LayerJSON, Size, Size2D } from "../../types.ts";
import { CPUMatrix } from "../matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Convolutional layer.
 */
export class ConvCPULayer {
  outputSize!: Size2D;
  padding: number;
  stride: number;

  input!: CPUMatrix;
  kernel!: CPUMatrix;
  padded!: CPUMatrix;
  output!: CPUMatrix;

  constructor(config: ConvLayerConfig) {
    this.kernel = new CPUMatrix(
      config.kernel,
      config.kernelSize.x,
      config.kernelSize.y,
    );
    this.padding = config.padding || 0;
    this.stride = config.stride || 1;
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Size, _batches: number) {
    const wp = (inputSize as Size2D).x + 2 * this.padding;
    const hp = (inputSize as Size2D).y + 2 * this.padding;
    if (this.padding > 0) {
      this.padded = CPUMatrix.with(wp, hp);
      this.padded.fill(255)
    }
    const wo = 1 + Math.floor((wp - this.kernel.x) / this.stride);
    const ho = 1 + Math.floor((hp - this.kernel.y) / this.stride);
    this.output = CPUMatrix.with(wo, ho);
    this.outputSize = {x: wo, y: ho}
  }

  feedForward(input: CPUMatrix) {
    if (this.padding > 0) {
      for (let i = 0; i < input.x; i++) {
        for (let j = 0; j < input.y; j++) {
          const idx = this.padded.x * (this.padding + j) + this.padding + i;
          this.padded.data[idx] = input.data[j * input.x + i];
        }
      }
    } else {
      this.padded = input;
    }
    for (let i = 0; i < this.output.x; i++) {
      for (let j = 0; j < this.output.y; j++) {
        let sum = 0;
        for (let x = 0; x < this.kernel.x; x++) {
          for (let y = 0; y < this.kernel.y; y++) {
            const k = this.padded.x * (j * this.stride + y) +
              (i * this.stride + x);
            const l = y * this.kernel.x + x;
            sum += this.padded.data[k] * this.kernel.data[l];
          }
        }
        this.output.data[j * this.output.x + i] = sum;
      }
    }
    return this.output;
  }

  backPropagate(_error: CPUMatrix, _rate: number) {
  }

  toJSON(): LayerJSON {
    return {
      outputSize: this.outputSize,
      type: "conv"
    };
  }
}
