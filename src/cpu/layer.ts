import { DataArray, DataArrayConstructor, DataType } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { CPUActivationFn, LeakyRelu, Relu, Sigmoid, Tanh } from "./activation.ts";

export class CPULayer<T extends DataType> {
    public outputSize: number
    public weights!: Float32Array
    public activationFn: CPUActivationFn = new Sigmoid()

    constructor(config: LayerConfig) {
        this.outputSize = config.size
        this.setActivation(config.activation)
    }

    public setActivation(activation: Activation) {
        switch (activation) {
            case Activation.Sigmoid:
                this.activationFn = new Sigmoid();
                break
            case Activation.LeakyRelu:
                this.activationFn = new LeakyRelu();
                break
            case Activation.Tanh:
                this.activationFn = new Tanh();
                break
            case Activation.Relu:
                this.activationFn = new Relu();
                break
        }
    }

    public feedForward(
        input: DataArray<T>, 
        type: DataType, 
        batches: number,
        inputSize: number
    ): DataArray<T> {
        const output = new DataArrayConstructor[type](this.outputSize * batches)
        
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
}