export type Shape1D = [number];

export type Shape2D = [number, number];

export type Shape3D = [number, number, number];

export type Shape4D = [number, number, number, number];

export type Shape5D = [number, number, number, number, number];

export type Shape6D = [number, number, number, number, number, number];

export enum Rank {
  R1 = 1, // Scalar   (magnitude only)
  R2 = 2, // Vector   (magnitude and direction)
  R3 = 3, // Matrix   (table of numbers)
  R4 = 4, // 3-Tensor (cube of numbers)
  R5 = 5,
  R6 = 6,
}

export interface Shape {
  1: Shape1D;
  2: Shape2D;
  3: Shape3D;
  4: Shape4D;
  5: Shape5D;
  6: Shape6D;
}

export type ArrayMap = Array1D | Array2D | Array3D | Array4D | Array5D | Array6D;

export type Array1D = number[];

export type Array2D = number[][];

export type Array3D = number[][][];

export type Array4D = number[][][][];

export type Array5D = number[][][][][];

export type Array6D = number[][][][][][];
