import { DataArray, DataArrayConstructor, DataType } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { CPUActivationFn, LeakyRelu, Relu, Sigmoid, Tanh } from "./activation.ts";
import { CPUCostFunction, CrossEntropy } from "./cost.ts";

export class CPULayer<T extends DataType = DataType> {
    public outputSize: number
    public activationFn: CPUActivationFn = new Sigmoid()
    public costFunction: CPUCostFunction = new CrossEntropy()

    public inputSize!: number
    public batches!: number
    
    public weights!: DataArray<T>
    public product!: DataArray<T>
    public output!: DataArray<T>
    public weightsDelta!: DataArray<T>
    public biasesDelta!: DataArray<T>

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

    public feedForward(input: DataArray<T>): DataArray<T> {
        for (let x = 0; x < this.outputSize; x++) {
            for (let y = 0; y < this.batches; y++) {
                let weightedSum = 0
                for (let k = 0; k < this.inputSize; k++) {
                    const a = k + y * this.inputSize;
                    const b = x + k * this.outputSize;  
                    weightedSum += input[a] * this.weights[b];
                    
                }
                const idx = x + y * this.outputSize;
                this.product[idx] = weightedSum;
                this.output[idx] = this.activationFn.activate(weightedSum);
            }
        }

        return this.output
    }

    public cost(input: DataArray<T>, output: DataArray<T>): DataArray<T> {
        for (let i = 0; i < input.length; i++) {
            this.biasesDelta[i] = this.costFunction.measure(this.product[i], input[i], output[i]);
        }
        return input;
    }


    public backPropagate() {
        
    }

    public initialize(type: DataType, inputSize: number, batches: number) {
        this.inputSize = inputSize
        const weightsLength = this.outputSize * inputSize
        const outputLength = this.outputSize * batches
        this.weights = new DataArrayConstructor[type](weightsLength) as DataArray<T>
        this.output = new DataArrayConstructor[type](outputLength) as DataArray<T>
        this.product = new DataArrayConstructor[type](outputLength) as DataArray<T>
        this.weights.fill(1)
    }
}