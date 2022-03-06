import { DataArray, DataType } from "../deps.ts";

export interface Network<T extends DataType = DataType> {
    addLayers(layer: LayerConfig[]): void
    getOutput(): DataArray<T>
    train(datasets: DataSet[], epochs: number, batches: number, learningRate: number): void
    predict(input: DataArray<T>): DataArray<T>
}

export interface NetworkConfig {
    input?: InputConfig;
    hidden: LayerConfig[];
    cost: Cost;
    output: LayerConfig;
}

export interface LayerConfig {
    size: number
    activation: Activation
}

export type Activation = "sigmoid" | "tanh" | "relu" | "leakyrelu"

export type Cost = "crossentropy" | "hinge"

export type Shape = number

export type InputConfig = {
    size?: number,
    type: DataType
}

export type NumberArray<T extends DataType = DataType>
    = DataArray<T> | Array<number>

export type DataSet = {
    inputs: NumberArray
    outputs: NumberArray
}