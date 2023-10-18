/**
 * Line Type for Jupyter Notebook
 */
export interface Line {
  x: number[];
  y: number[];
  type?: "scatter" | "bar";
  mode?: "markers" | "lines" | "lines+markers";
  name?: string;
  line?: {
    color?: string;
    width?: number;
  };
}
