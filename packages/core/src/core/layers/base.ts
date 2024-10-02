export class Layer {
  outputShape: bigint[];
  inLayers: Layer[];
  outLayers: Layer[];
  constructor(outputShape: (number | bigint)[]) {
    this.outputShape = outputShape.map(BigInt);
    this.inLayers = [];
    this.outLayers = [];
  }
  /**
   * Connect this layer to the output of another layer.
   * @param layer Layer preceedind this layer.
   * @returns 
   */
  connect(layer: Layer): typeof this {
    this.in(layer);
    layer.out(this);
    return this;
  }
  in(layer: Layer): typeof this {
    this.inLayers.push(layer);
    return this;
  }
  out(layer: Layer): typeof this {
    this.outLayers.push(layer);
    return this;
  }
}
