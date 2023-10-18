/**
 * Line Type for Jupyter Notebook
 */
export interface Line {
    x: number[];
    y: number[];
    type: "scatter";
    name: string;
    line: {
        color: string;
        width: number;
    }
}