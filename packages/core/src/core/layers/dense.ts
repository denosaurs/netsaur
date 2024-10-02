import { LayerType, Init } from "../types.ts";
import type { DenseLayerConfig } from "../api/layer.ts";
import lib from "../ffi.ts";
import {
  Matrix,
  Tensor,
  type MatrixLike,
} from "../../../../tensor/mod.ts";
import { Layer } from "./base.ts";

export class DenseLayer extends Layer implements DenseLayerConfig {
  backend?: bigint;
  size: number;
  init: Init;
  c: number;
  l1Ratio: number;
  inputSize: number;
  input?: Tensor<"f32", 2>;
  constructor(config: DenseLayerConfig, inputSize = 0) {
    super([config.size]);
    this.size = config.size;
    this.inputSize = inputSize;
    this.c = config.c ?? 0;
    this.l1Ratio = config.l1Ratio ?? 1;
    this.init = config.init ?? Init.Uniform;
  }
  create() {
    const layer: { type: LayerType; config: DenseLayerConfig } = {
      type: LayerType.Dense,
      config: {
        c: this.c,
        l1Ratio: this.l1Ratio,
        size: this.size,
        init: this.init,
      },
    };
    const serialized = JSON.stringify(layer);
    const buffer = new TextEncoder().encode(serialized);
    this.backend = lib.symbols.ffi_layer_create(
      buffer,
      BigInt(buffer.length),
      BigUint64Array.from([1n, ...this.inLayers[0].outputShape]),
      BigInt(this.inLayers[0].outputShape.length + 1)
    );
  }
  forward(data: MatrixLike<"f32">, training = false): Matrix<"f32"> {
    if (!this.backend)
      throw new Error(
        `Layer not initialized. Use ${this.constructor.name}.create() first!`
      );
    this.input = new Tensor(data);
    const res = new Matrix("f32", data.shape);
    const shape = BigInt64Array.from(data.shape.map((x) => BigInt(x)));
    lib.symbols.ffi_layer_forward(
      this.backend,
      new Uint8Array(data.data.buffer),
      new Uint8Array(shape.buffer),
      BigInt(shape.length),
      new Uint8Array(res.data.buffer),
      training
    );
    return res;
  }
  backward(data: MatrixLike<"f32">): Matrix<"f32"> {
    if (!this.backend)
      throw new Error(
        `Layer not initialized. Use ${this.constructor.name}.create() first!`
      );
    const res = new Matrix("f32", data.shape);
    const shape = BigInt64Array.from(data.shape.map((x) => BigInt(x)));
    lib.symbols.ffi_layer_backward(
      this.backend,
      new Uint8Array(data.data.buffer),
      new Uint8Array(shape.buffer),
      BigInt(shape.length),
      new Uint8Array(res.data.buffer)
    );
    return res;
  }
}
