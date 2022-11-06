import { GPUInstance } from "../backends/gpu/mod.ts";
import { WebGPUData } from "../deps.ts";
import {
  BackendType,
  Rank,
  Shape,
  TensorData,
  TensorJSON,
  TensorLike,
  TypedArray,
} from "./types.ts";
import {
  flatten,
  inferShape,
  iterate1D,
  Random,
  to1D,
  to2D,
  to3D,
  toShape,
} from "./util.ts";

// NOTE: In development, avoid calling boilerplate-heavy methods such as Tensor.zeroes()
// Such methods should be reserved as user-facing API
// Use more specific albeit verbous functions such cpuZeroes2D()

export class Tensor<R extends Rank, B extends BackendType> {
  static type: BackendType;
  shape: Shape[R];
  data: TensorData[B];

  get x() {
    return this.shape[0];
  }

  get y() {
    return this.shape[1] || 1;
  }

  get z() {
    return this.shape[2] || 1;
  }

  get w() {
    return this.shape[3] || 1;
  }

  constructor(data: TensorData[B], shape: Shape[R]) {
    this.shape = shape;
    this.data = data;
  }

  static zeroes<R extends Rank, B extends BackendType>(
    shape: Shape[R],
  ): Tensor<R, B> {
    const data = new Float32Array(to1D(shape)[0]).fill(0);
    return new Tensor(toData(data), shape);
  }

  static ones<R extends Rank, B extends BackendType>(
    shape: Shape[R],
    value = 1,
  ): Tensor<R, B> {
    const data = new Float32Array(to1D(shape)[0]).fill(value);
    return new Tensor(toData(data), shape);
  }

  static randomNormal<R extends Rank, B extends BackendType>(
    shape: Shape[R],
    mean = 0,
    stdDev = 1,
    seed?: number,
  ): Tensor<R, B> {
    if (seed !== undefined) Random.setSeed(seed);
    const data = new Float32Array(to1D(shape)[0]).fill(
      Random.gaussian(mean, stdDev),
    );
    return new Tensor(toData(data), shape);
  }

  to1D(): Tensor<Rank.R1, B> {
    return new Tensor(this.data, to1D(this.shape));
  }

  to2D(): Tensor<Rank.R2, B> {
    return new Tensor(this.data, to2D(this.shape));
  }

  to3D(): Tensor<Rank.R3, B> {
    return new Tensor(this.data, to3D(this.shape));
  }

  index(...indices: number[]) {
    // let index = indices[0];
    // for (let i = 1; i < indices.length; i++) {
    //   index += indices[i] * this.shape[i - 1];
    // }
    // return index
    switch (indices.length) {
      case 2:
        return indices[0] + indices[1] * this.x;
      case 3:
        return indices[0] + indices[1] * this.x + indices[2] * this.x * this.y;
      default:
        return indices[0] + indices[1] * this.x + indices[2] * this.x * this.y + indices[3] * this.x * this.y * this.z;
    }

  }

  async get(...indices: number[]) {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * this.shape[i];
    }
    return (await this.getData())[index];
  }

  async getData() {
    switch (Tensor.type) {
      case BackendType.CPU:
      case BackendType.Native:
        return Array.from(this.data as TypedArray);
      case BackendType.GPU: {
        const data = await (this.data as WebGPUData).get();
        return Array.from(data);
      }
    }
  }

  setData(values: Float32Array) {
    switch (Tensor.type) {
      case BackendType.CPU:
        return (this.data as Float32Array).set(values);
      case BackendType.GPU: {
        const data = (this.data as TensorData[BackendType.GPU]).buffer;
        GPUInstance.backend!.device.queue.writeBuffer(data, 0, values);
      }
    }
  }

  async toJSON() {
    return {
      data: await this.getData(),
      shape: this.shape,
    };
  }

  static fromJSON<R extends Rank, B extends BackendType>(
    tensor: TensorJSON,
  ): Tensor<R, B> {
    return new Tensor(toData(tensor.data), tensor.shape as Shape[R]);
  }

  fmt() {
    const data = this.data as TensorData[BackendType.CPU];
    let res = "Tensor [\n";
    iterate1D(this.y, (i: number) => {
      res += "  [ ";
      const row = data.slice(i * this.x, (i + 1) * this.x);
      iterate1D(row.length, (j: number) => {
        res += row[j].toString() + ", ";
      });
      res += "]\n";
    });
    res += "]";
    return res;
  }
}

export function toData<B extends BackendType>(
  values: TensorLike,
): TensorData[B] {
  switch (Tensor.type) {
    case BackendType.Native:
    case BackendType.CPU:
      return new Float32Array(flatten(values)) as TensorData[B];
    case BackendType.GPU: {
      const data = new Float32Array(flatten(values));
      const res = new WebGPUData(GPUInstance.backend!, "f32", data.length);
      GPUInstance.backend!.device.queue.writeBuffer(res.buffer, 0, data);
      return res as TensorData[B];
    }
  }
}

export function tensor1D(
  values: TensorLike,
  shape?: Shape[Rank.R1],
) {
  // deno-lint-ignore no-explicit-any
  if (Array.isArray((values as any)[0])) throw new Error("Invalid 1D Tensor");
  const outputShape = shape || [(values as TypedArray).length];
  return new Tensor(toData(values), outputShape);
}

export function zeros1D(shape: Shape[Rank.R1]) {
  return Tensor.zeroes(shape);
}

export function randNormal1D(
  shape: Shape[Rank.R1],
  mean = 0,
  stdDev = 1,
  seed?: number,
) {
  return Tensor.randomNormal(shape, mean, stdDev, seed);
}

export function tensor2D(
  values: TensorLike,
  shape?: Shape[Rank.R2],
) {
  const outputShape = shape || inferShape(values).slice();
  if (outputShape.length > 2) throw new Error("Invalid 2D Tensor");
  // values
  return new Tensor(toData(values), [outputShape[1], outputShape[0]]);
}

export function zeros2D(shape: Shape[Rank.R2]) {
  return Tensor.zeroes([shape[1], shape[0]]);
}

export function randNormal2D(
  shape: Shape[Rank.R2],
  mean = 0,
  stdDev = 1,
  seed?: number,
) {
  return Tensor.randomNormal([shape[1], shape[0]], mean, stdDev, seed);
}

export function tensor3D(
  values: TensorLike,
  shape?: Shape[Rank.R3],
) {
  const outputShape = shape || inferShape(values).slice();
  if (outputShape.length > 3) throw new Error("Invalid 3D Tensor");
  // values
  return new Tensor(toData(values), [
    outputShape[2],
    outputShape[1],
    outputShape[0],
  ]);
}

export function zeros3D(shape: Shape[Rank.R3]) {
  return Tensor.zeroes([shape[2], shape[1], shape[0]]);
}

export function randNormal3D(
  shape: Shape[Rank.R3],
  mean = 0,
  stdDev = 1,
  seed?: number,
) {
  return Tensor.randomNormal(
    [shape[2], shape[1], shape[0]],
    mean,
    stdDev,
    seed,
  );
}

export function tensor4D(
  values: TensorLike,
  shape?: Shape[Rank.R4],
) {
  const outputShape = shape || inferShape(values).slice();
  if (outputShape.length > 4) throw new Error("Invalid 4D Tensor");
  // values
  return new Tensor(toData(values), [
    outputShape[3],
    outputShape[2],
    outputShape[1],
    outputShape[0],
  ]);
}

export function zeros4D(shape: Shape[Rank.R4]) {
  return Tensor.zeroes([shape[3], shape[2], shape[1], shape[0]]);
}

export function randNormal4D(
  shape: Shape[Rank.R4],
  mean = 0,
  stdDev = 1,
  seed?: number,
) {
  return Tensor.randomNormal(
    [shape[3], shape[2], shape[1], shape[0]],
    mean,
    stdDev,
    seed,
  );
}

export function tensor5D(
  values: TensorLike,
  shape?: Shape[Rank.R5],
) {
  const outputShape = shape || inferShape(values).slice();
  if (outputShape.length > 5) throw new Error("Invalid 5D Tensor");
  // values
  return new Tensor(toData(values), [
    outputShape[4],
    outputShape[3],
    outputShape[2],
    outputShape[1],
    outputShape[0],
  ]);
}

export function zeros5D(shape: Shape[Rank.R5]) {
  return Tensor.zeroes([shape[4], shape[3], shape[2], shape[1], shape[0]]);
}

export function randNormal5D(
  shape: Shape[Rank.R5],
  mean = 0,
  stdDev = 1,
  seed?: number,
) {
  return Tensor.randomNormal(
    [shape[4], shape[3], shape[2], shape[1], shape[0]],
    mean,
    stdDev,
    seed,
  );
}

export function tensor6D(
  values: TensorLike,
  shape?: Shape[Rank.R6],
) {
  const outputShape = shape || inferShape(values).slice();
  if (outputShape.length > 6) throw new Error("Invalid 6D Tensor");
  // values
  return new Tensor(toData(values), [
    outputShape[5],
    outputShape[4],
    outputShape[3],
    outputShape[2],
    outputShape[1],
    outputShape[0],
  ]);
}

export function zeros6D(shape: Shape[Rank.R6]) {
  return Tensor.zeroes([
    shape[5],
    shape[4],
    shape[3],
    shape[2],
    shape[1],
    shape[0],
  ]);
}

export function randNormal6D(
  shape: Shape[Rank.R6],
  mean = 0,
  stdDev = 1,
  seed?: number,
) {
  return Tensor.randomNormal(
    [shape[5], shape[4], shape[3], shape[2], shape[1], shape[0]],
    mean,
    stdDev,
    seed,
  );
}

export function cpuZeroes2D(
  shape: Shape[Rank.R2],
): Tensor<Rank.R2, BackendType.CPU> {
  const data = new Float32Array(shape[0] * shape[1]);
  return new Tensor(data.fill(0), shape);
}

export function cpuZeroes3D(
  shape: Shape[Rank.R3],
): Tensor<Rank.R3, BackendType.CPU> {
  const data = new Float32Array(shape[0] * shape[1] * shape[2]);
  return new Tensor(data.fill(0), shape);
}

export function gpuZeroes2D(
  shape: Shape[Rank.R2],
): Tensor<Rank.R2, BackendType.GPU> {
  const data = new Float32Array(shape[0] * shape[1]);
  const res = new WebGPUData(GPUInstance.backend!, "f32", data.length);
  GPUInstance.backend!.device.queue.writeBuffer(res.buffer, 0, data);
  return new Tensor(res, shape);
}

export function toShape2D(shape: Shape[Rank]): Shape[Rank.R2] {
  return toShape(shape, Rank.R2);
}

export function toShape3D(shape: Shape[Rank]): Shape[Rank.R3] {
  return toShape(shape, Rank.R3);
}

export function reshape2D<B extends BackendType>(tensor: Tensor<Rank, B>) {
  const res = new Tensor(tensor.data, toShape2D(tensor.shape));
  return res as Tensor<Rank.R2, B>;
}

export function reshape3D<B extends BackendType>(tensor: Tensor<Rank, B>) {
  const res = new Tensor(tensor.data, toShape3D(tensor.shape));
  return res as Tensor<Rank.R3, B>;
}

export function cpuCloneZeroes(
  tensor: Tensor<Rank, BackendType.CPU>,
): Tensor<Rank, BackendType.CPU> {
  const data = new Float32Array(tensor.data.length);
  return new Tensor(data, tensor.shape);
}

export function cpuZeroes1D(
  shape: Shape[Rank.R1],
): Tensor<Rank.R1, BackendType.CPU> {
  const data = new Float32Array(shape[0]);
  return new Tensor(data.fill(0), shape);
}

export function cpuZeroes4D(
  shape: Shape[Rank.R4],
): Tensor<Rank.R4, BackendType.CPU> {
  const data = new Float32Array(shape[0] * shape[1] * shape[2] * shape[3]);
  return new Tensor(data.fill(0), shape);
}

export function toShape4D(shape: Shape[Rank]): Shape[Rank.R4] {
  return toShape(shape, Rank.R4);
}

export function reshape4D<B extends BackendType>(tensor: Tensor<Rank, B>) {
  const res = new Tensor(tensor.data, toShape4D(tensor.shape));
  return res as Tensor<Rank.R4, B>;
}
