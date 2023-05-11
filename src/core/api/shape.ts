/**
 * 1st dimentional shape.
 */
export type Shape1D = [number];

/**
 * 2nd dimentional shape.
 */
export type Shape2D = [number, number];

/**
 * 3th dimentional shape.
 */
export type Shape3D = [number, number, number];

/**
 * 4th dimentional shape.
 */
export type Shape4D = [number, number, number, number];

/**
 * 5th dimentional shape.
 */
export type Shape5D = [number, number, number, number, number];

/**
 * 6th dimentional shape.
 */
export type Shape6D = [number, number, number, number, number, number];

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
 * Shape Interface
 */
export interface Shape {
  1: Shape1D;
  2: Shape2D;
  3: Shape3D;
  4: Shape4D;
  5: Shape5D;
  6: Shape6D;
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
