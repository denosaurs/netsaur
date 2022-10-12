import {
  ConvLayerConfig,
  LayerJSON,
  MatrixJSON,
  Size,
  Size2D,
} from "../../../core/types.ts";
import { iterate2D, to2D } from "../../../core/util.ts";
import { CPUMatrix } from "../matrix.ts";

// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

/**
 * Convolutional layer.
 */
export class ConvCPULayer {
  outputSize!: Size2D;
  padding: number;
  strides: Size2D;

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
    this.strides = to2D(config.strides);
  }

  reset(_batches: number) {
  }

  initialize(inputSize: Size, _batches: number) {
    const wp = (inputSize as Size2D).x + 2 * this.padding;
    const hp = (inputSize as Size2D).y + 2 * this.padding;
    if (this.padding > 0) {
      this.padded = CPUMatrix.with(wp, hp);
      this.padded.fill(255);
    }
    const wo = 1 + Math.floor((wp - this.kernel.x) / this.strides.x);
    const ho = 1 + Math.floor((hp - this.kernel.y) / this.strides.y);
    this.output = CPUMatrix.with(wo, ho);
    this.outputSize = { x: wo, y: ho };
  }

  feedForward(input: CPUMatrix): CPUMatrix {
    if (this.padding > 0) {
      iterate2D(input, (i: number, j: number) => {
        const idx = this.padded.x * (this.padding + j) + this.padding + i;
        this.padded.data[idx] = input.data[j * input.x + i];
      });
    } else {
      this.padded = input;
    }
    iterate2D(this.output, (i: number, j: number) => {
      let sum = 0;
      iterate2D(this.kernel, (x: number, y: number) => {
        const k = this.padded.x * (j * this.strides.y + y) + (i * this.strides.x + x);
        const l = y * this.kernel.x + x;
        sum += this.padded.data[k] * this.kernel.data[l];
      });
      this.output.data[j * this.output.x + i] = sum;
    });
    return this.output;
  }

  backPropagate(_error: CPUMatrix, _rate: number) {
  }

  toJSON(): LayerJSON {
    return {
      outputSize: this.outputSize,
      type: "conv",
      input: this.input.toJSON(),
      kernel: this.kernel.toJSON(),
      padded: this.padded.toJSON(),
      output: this.output.toJSON(),
      strides: this.strides,
      padding: this.padding,
    };
  }

  static fromJSON(
    { outputSize, input, kernel, padded, output, strides, padding }: LayerJSON,
  ): ConvCPULayer {
    const layer = new ConvCPULayer({
      kernel: (kernel as MatrixJSON).data,
      kernelSize: { x: (kernel as MatrixJSON).x, y: (kernel as MatrixJSON).y },
      padding,
      strides,
    });
    layer.input = new CPUMatrix(input.data, input.x, input.y);
    layer.outputSize = outputSize as Size2D;
    layer.padded = new CPUMatrix(
      (padded as MatrixJSON).data,
      (padded as MatrixJSON).x,
      (padded as MatrixJSON).y,
    );
    layer.output = new CPUMatrix(output.data, output.x, output.y);
    return layer;
  }
}
