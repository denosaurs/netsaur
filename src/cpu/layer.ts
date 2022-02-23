import { DataType } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { CPUActivationFn, LeakyRelu, Relu, Sigmoid, Tanh } from "./activation.ts";
import { CPUCostFunction, CrossEntropy } from "./cost.ts";
import { CPUMatrix } from "./matrix.ts";

export class CPULayer<T extends DataType = DataType> {
    public outputSize: number
    public activationFn: CPUActivationFn = new Sigmoid()
    public costFunction: CPUCostFunction = new CrossEntropy()
    
    public weights!: CPUMatrix
    public product!: CPUMatrix
    public output!: CPUMatrix
    public weightsDelta!: CPUMatrix
    public biasesDelta!: CPUMatrix

    constructor(config: LayerConfig) {
        this.outputSize = config.size
        this.setActivation(config.activation)
    }

    public initialize(type: DataType, inputSize: number, batches: number) {
        this.weights = CPUMatrix.with(this.outputSize, inputSize, type)
        this.output = CPUMatrix.with(this.outputSize, batches, type)
        this.product = CPUMatrix.with(this.outputSize, batches, type)
        this.weights.fill(1)
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

    public feedForward(input: CPUMatrix): CPUMatrix {
        this.product = CPUMatrix.mul(input, this.weights)
        for (let i = 0; i < this.product.data.length; i++) {
            this.output.data[i] = this.activationFn.activate(this.product.data[i]);
        }
        return this.output
    }

    public cost(input: CPUMatrix, output: CPUMatrix): CPUMatrix {
        for (let i = 0; i < input.data.length; i++) {
            this.biasesDelta.data[i] = this.costFunction.measure(
                this.product.data[i], 
                input.data[i], 
                output.data[i]
            );
        }
        return input;
    }


    public backPropagate() {
        
    }
}