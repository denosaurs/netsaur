/**
 * Shape Type
 */
export type Shape<R extends Rank> = [number, ...number[]] & { length: R };
/**
 * 1st dimentional shape.
 */
export type Shape1D = Shape<1>;

/**
 * 2nd dimentional shape.
 */
export type Shape2D = Shape<2>;

/**
 * 3th dimentional shape.
 */
export type Shape3D = Shape<3>;

/**
 * 4th dimentional shape.
 */
export type Shape4D = Shape<4>;

/**
 * 5th dimentional shape.
 */
export type Shape5D = Shape<5>;

/**
 * 6th dimentional shape.
 */
export type Shape6D = Shape<6>;

/**
 * Rank Types.
 */
export enum Rank {
  /**
   * Scalar   (magnitude only).
   */
  R1 = 1,

  /**
   * Vector   (magnitude and direction).
   */
  R2 = 2,

  /**
   * Matrix   (table of numbers).
   */
  R3 = 3,

  /**
   *  3-Tensor (cube of numbers)
   */
  R4 = 4,

  /**
   * Rank 5 Tensor
   */
  R5 = 5,

  /**
   * Rank 6 Tensor
   */
  R6 = 6,
}

/**
 * Array Map Types.
 */
export type ArrayMap =
  | Array1D
  | Array2D
  | Array3D
  | Array4D
  | Array5D
  | Array6D;

/**
 * 1D Array.
 */
export type Array1D = number[];

/**
 * 2D Array.
 */
export type Array2D = number[][];

/**
 * 3D Array.
 */
export type Array3D = number[][][];

/**
 * 4D Array.
 */
export type Array4D = number[][][][];

/**
 * 5D Array.
 */
export type Array5D = number[][][][][];

/**
 * 6D Array.
 */
export type Array6D = number[][][][][][];
