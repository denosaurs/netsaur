import ffi from "./ffi.ts";

const {
  matrix_new,
  matrix_new_randf,
  matrix_new_from_array,
  matrix_new_fill_f32,
  matrix_new_fill_i32,
  matrix_new_fill_u32,
  matrix_free,
  matrix_dot,
  matrix_add,
  matrix_sub,
  matrix_add_f32,
  matrix_add_i32,
  matrix_add_u32,
  matrix_div_f32,
  matrix_div_i32,
  matrix_div_u32,
  matrix_copy,
  matrix_mul_f32,
  matrix_mul_i32,
  matrix_mul_u32,
  matrix_sub_f32,
  matrix_sub_i32,
  matrix_sub_u32,
  matrix_transpose,
} = ffi;

export type DataType = "u32" | "i32" | "f32";

enum C_DATA_TYPE {
  u32 = 0,
  i32 = 1,
  f32 = 2,
}

function typeFromArray(array: Float32Array | Int32Array | Uint32Array) {
  if (array instanceof Float32Array) return C_DATA_TYPE.f32;
  if (array instanceof Int32Array) return C_DATA_TYPE.i32;
  if (array instanceof Uint32Array) return C_DATA_TYPE.u32;
  throw new Error("Unsupported array type");
}

export type DataTypeArray<T extends DataType> = T extends "u32" ? Uint32Array
  : T extends "i32" ? Int32Array
  : T extends "f32" ? Float32Array
  : never;

const {
  op_ffi_read_u64,
  op_ffi_read_u32,
  op_ffi_read_u8,
  op_ffi_get_buf,
  // deno-lint-ignore no-explicit-any
} = (Deno as any).core.ops;

const MatrixFinalizer = new FinalizationRegistry((id: Deno.PointerValue) => {
  matrix_free(id);
});

export class Matrix<T extends DataType> {
  #ptr: Deno.PointerValue = 0;
  #token = { ptr: 0 as Deno.PointerValue };

  get unsafePointer() {
    return this.#ptr;
  }

  declare readonly type: T;
  declare readonly rows: number;
  declare readonly cols: number;
  declare readonly data: DataTypeArray<T>;

  constructor(ptr: Deno.PointerValue);
  constructor(rows: number, cols: number, type: T);
  constructor(rows: number, cols: number, data: DataTypeArray<T>);
  constructor(rows: number, cols: number, type: T, fill: number);
  constructor(
    rows: number,
    cols?: number,
    type?: T | DataTypeArray<T>,
    fill?: number | Deno.PointerValue,
  ) {
    if (cols === undefined) {
      this.#ptr = rows as Deno.PointerValue;
    } else if (typeof type === "string" && type in C_DATA_TYPE) {
      if (typeof fill === "number") {
        switch (type) {
          case "u32":
            this.#ptr = matrix_new_fill_u32(rows, cols!, fill);
            break;
          case "i32":
            this.#ptr = matrix_new_fill_i32(rows, cols!, fill);
            break;
          case "f32":
            this.#ptr = matrix_new_fill_f32(rows, cols!, fill);
            break;
        }
      } else {
        this.#ptr = matrix_new(rows, cols!, C_DATA_TYPE[type]);
      }
      Object.defineProperty(this, "type", { value: type, writable: false });
    } else if (typeof type === "object" && type !== null) {
      const ty = typeFromArray(type);
      Object.defineProperty(this, "type", {
        value: C_DATA_TYPE[ty],
        writable: false,
      });
      this.#ptr = matrix_new_from_array(rows, cols!, ty, type);
    } else throw new Error("Invalid arguments");

    const c = Number(this.#ptr);

    const ptr = op_ffi_read_u64(c);
    rows = op_ffi_read_u32(c + 8);
    cols = op_ffi_read_u32(c + 12);

    if (!("type" in this)) {
      const ty = op_ffi_read_u8(c + 16);
      Object.defineProperty(this, "type", {
        value: C_DATA_TYPE[ty],
        writable: false,
      });
    }

    const buf = op_ffi_get_buf(ptr, rows * cols! * 4);

    Object.defineProperties(this, {
      rows: { value: rows, writable: false },
      cols: { value: cols, writable: false },
      data: {
        value: this.type === "f32"
          ? new Float32Array(buf)
          : this.type === "i32"
          ? new Int32Array(buf)
          : new Uint32Array(buf),
        writable: false,
      },
    });

    this.#token.ptr = this.#ptr;
    MatrixFinalizer.register(this, this.#ptr, this.#token);
  }

  static rand(rows: number, cols: number): Matrix<"f32"> {
    return new Matrix(matrix_new_randf(rows, cols));
  }

  static of(data: number[][]): Matrix<"f32"> {
    const rows = data.length;
    const cols = data[0].length;
    return new Matrix(rows, cols, new Float32Array(data.flat()));
  }

  static row(data: number[]): Matrix<"f32"> {
    return new Matrix(1, data.length, new Float32Array(data));
  }

  static column(data: number[]): Matrix<"f32"> {
    return new Matrix(data.length, 1, new Float32Array(data));
  }

  dot(b: Matrix<T>): Matrix<T> {
    const c = matrix_dot(this.#ptr, b.unsafePointer);
    if (c === 0) throw new Error("Invalid matrix dimensions");
    return new Matrix(c);
  }

  add(b: Matrix<T> | number): Matrix<T> {
    if (typeof b === "number") {
      switch (this.type) {
        case "u32":
          return new Matrix(matrix_add_u32(this.#ptr, b));
        case "i32":
          return new Matrix(matrix_add_i32(this.#ptr, b));
        case "f32":
          return new Matrix(matrix_add_f32(this.#ptr, b));
        default:
          throw new Error("unreachable");
      }
    } else {
      const c = matrix_add(this.#ptr, b.unsafePointer);
      if (c === 0) throw new Error("Invalid matrix dimensions");
      return new Matrix(c);
    }
  }

  sub(b: Matrix<T> | number): Matrix<T> {
    if (typeof b === "number") {
      switch (this.type) {
        case "u32":
          return new Matrix(matrix_sub_u32(this.#ptr, b));
        case "i32":
          return new Matrix(matrix_sub_i32(this.#ptr, b));
        case "f32":
          return new Matrix(matrix_sub_f32(this.#ptr, b));
        default:
          throw new Error("unreachable");
      }
    } else {
      const c = matrix_sub(this.#ptr, b.unsafePointer);
      if (c === 0) throw new Error("Invalid matrix dimensions");
      return new Matrix(c);
    }
  }

  mul(b: Matrix<T> | number): Matrix<T> {
    if (typeof b === "number") {
      switch (this.type) {
        case "u32":
          return new Matrix(matrix_mul_u32(this.#ptr, b));
        case "i32":
          return new Matrix(matrix_mul_i32(this.#ptr, b));
        case "f32":
          return new Matrix(matrix_mul_f32(this.#ptr, b));
        default:
          throw new Error("unreachable");
      }
    } else {
      return this.dot(b);
    }
  }

  div(b: number): Matrix<T> {
    switch (this.type) {
      case "u32":
        return new Matrix(matrix_div_u32(this.#ptr, b));
      case "i32":
        return new Matrix(matrix_div_i32(this.#ptr, b));
      case "f32":
        return new Matrix(matrix_div_f32(this.#ptr, b));
      default:
        throw new Error("unreachable");
    }
  }

  transpose(): Matrix<T> {
    return new Matrix(matrix_transpose(this.#ptr));
  }

  copy(): Matrix<T> {
    return new Matrix(matrix_copy(this.#ptr));
  }

  free() {
    if (this.#ptr) {
      matrix_free(this.#ptr);
      MatrixFinalizer.unregister(this.#token);
      this.#ptr = 0;
    }
  }

  [Symbol.for("Deno.customInspect")]() {
    let result = `Matrix<${this.type}>(${this.rows}x${this.cols}) [\n`;
    for (let i = 0; i < this.rows; i++) {
      result += "  [" +
        this.data.subarray(i * this.cols, (i + 1) * this.cols).join(", ") +
        "]" + (i < this.rows - 1 ? "," : "") + "\n";
    }
    result += "]";
    return result;
  }
}
