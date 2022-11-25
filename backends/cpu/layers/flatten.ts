import { InvalidFlattenError } from "../../../core/error.ts";
import { Tensor } from "../../../core/tensor.ts";
import {
  CPUTensor,
  FlattenLayerConfig,
  LayerJSON,
  Rank,
  Shape,
} from "../../../core/types.ts";

export class FlattenCPULayer {
  type = "flatten";
  inputSize!: Shape[Rank];
  outputSize: Shape[Rank];
  output!: CPUTensor<Rank>

  constructor(config: FlattenLayerConfig) {
    this.outputSize = config.size;
  }

  reset(_: number) {}

  initialize(inputSize: Shape[Rank]) {
    if (
      inputSize.reduce((i, j) => i * j, 1) !=
        this.outputSize.reduce((i, j) => i * j, 1)
    ) {
      throw new InvalidFlattenError(inputSize, this.outputSize)
    }
    this.inputSize = inputSize.slice(0, -1) as Shape[Rank]
    const outputShape = [...this.outputSize, inputSize.at(-1)!] as Shape[Rank]
    this.output = new Tensor(new Float32Array(), outputShape);
  }

  feedForward(input: CPUTensor<Rank>): CPUTensor<Rank> {
    const batches = input.shape.at(-1)!
    return new Tensor(input.data, [...this.outputSize, batches] as Shape[Rank])
  }

  backPropagate(error: CPUTensor<Rank>, _rate: number) {
    const batches = error.shape.at(-1)!
    return new Tensor(error.data, [...this.inputSize, batches] as Shape[Rank])
  }

  // deno-lint-ignore require-await
  async toJSON() {
    return {
      outputSize: this.outputSize,
      inputSize: this.inputSize,
      type: this.type,
    };
  }

  static fromJSON(
    { inputSize, outputSize }: LayerJSON,
  ): FlattenCPULayer {
    const layer = new FlattenCPULayer({ size: outputSize as Shape[Rank] });
    layer.inputSize = inputSize!;
    return layer;
  }
}
