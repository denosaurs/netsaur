import { DataArray, DataArrayConstructor, DataType } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { CPUActivationFn, LeakyRelu, Relu, Sigmoid, Tanh } from "./activation.ts";

export class CPULayer<T extends DataType = DataType> {
    public outputSize: number
    public weights!: DataArray<T>
    public activationFn: CPUActivationFn = new Sigmoid()

    constructor(config: LayerConfig) {
        this.outputSize = config.size
        this.setActivation(config.activation)
    }

    public setActivation(activation: Activation) {
        switch (activation) {
            case "sigmoid":
                this.activationFn = new Sigmoid();
                break
            case "leakyrelu":
                this.activationFn = new LeakyRelu();
                break
            case "tanh":
                this.activationFn = new Tanh();
                break
            case "relu":
                this.activationFn = new Relu();
                break
        }
    }

    public feedForward(
        input: DataArray<T>, 
        batches: number, 
        inputSize: number, 
        type: DataType
    ): DataArray<T> {
        const output = new DataArrayConstructor[type](this.outputSize * batches)
        if (!this.weights) this.initialize(inputSize, type)
        
        for (let x = 0; x < this.outputSize; x++) {
            for (let y = 0; y < batches; y++) {
                let weightedSum = 0
                for (let k = 0; k < inputSize; k++) {
                    const a = k + y * inputSize;
                    const b = x + k * this.outputSize;  
                    weightedSum += input[a] * this.weights[b];
                    
                }
                const idx = x + y * this.outputSize;
                output[idx] = this.activationFn.activate(weightedSum);
            }
        }

        return output as DataArray<T>
    }

    // naive implementation
    public initialize(inputSize: number, type: DataType) {
        const length = this.outputSize * inputSize
        this.weights = new DataArrayConstructor[type](length) as DataArray<T>
        this.weights.fill(1)
    }
}