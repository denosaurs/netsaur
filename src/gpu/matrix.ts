import {
  DataType,
  DataTypeArray,
  WebGPUBackend,
  WebGPUData,
} from "../../deps.ts";
import { fromType } from "../util.ts";

export class GPUMatrix<T extends DataType = DataType> {
  constructor(
    public data: WebGPUData<T>,
    public x: number,
    public y: number,
    public type: DataType = data.type,
  ) {}

  static async with(
    backend: WebGPUBackend,
    x: number,
    y: number,
    type: DataType,
  ) {
    const data = new (fromType(type))(x * y);
    const buf = await WebGPUData.from(backend, data);
    return new this(buf, x, y, type);
  }

  static async from<T extends DataType = DataType>(
    backend: WebGPUBackend,
    data: DataTypeArray<T>,
    x: number,
    y: number,
    type?: DataType,
  ) {
    const buf = await WebGPUData.from(backend, data);
    return new this(buf, x, y, type);
  }
  toJSON() {
    return {
      data: this.data,
      x: this.x,
      y: this.y,
      type: this.type,
    };
  }
}
