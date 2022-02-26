import { DataType } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { CPUActivationFn, LeakyRelu, Relu, Sigmoid, Tanh } from "./activation.ts";
import { CPUCostFunction, CrossEntropy } from "./cost.ts";
import { CPUMatrix } from "./matrix.ts";

// this is pretty good too
// https://github.com/mnielsen/neural-networks-and-deep-learning
// https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#applying-the-chain-rule

export class CPULayer {
    public outputSize: number
    public activationFn: CPUActivationFn = new Sigmoid()
    public costFn: CPUCostFunction = new CrossEntropy()
    
    public input!: CPUMatrix
    public weights!: CPUMatrix
    public product!: CPUMatrix
    public biases!: CPUMatrix
    public output!: CPUMatrix
    public error!: CPUMatrix

    constructor(config: LayerConfig) {
        this.outputSize = config.size
        this.setActivation(config.activation)
    }

    public initialize(type: DataType, inputSize: number, batches: number) {
        this.weights = CPUMatrix.with(this.outputSize, inputSize, type)
        this.output = CPUMatrix.with(this.outputSize, batches, type)
        this.biases = CPUMatrix.with(this.outputSize, batches, type)
        this.product = CPUMatrix.with(this.outputSize, batches, type)
        this.error = CPUMatrix.with(this.outputSize, batches, type)
        this.biases.fill(1)
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
        this.input = input
        this.product = CPUMatrix.mul(input, this.weights)
        for (let i = 0; i < this.product.data.length; i++) {
            const activation = this.activationFn.activate(this.product.data[i])
            this.output.data[i] = activation + this.biases.data[i];
        }
        return this.output
    }

    public cost(input: CPUMatrix, output: CPUMatrix): CPUMatrix {
        for (let i = 0; i < input.data.length; i++) {
            this.error.data[i] = this.costFn.prime(
                this.product.data[i],
                output.data[i]
            );
        }
        return input;
    }

    public backPropagate(error: CPUMatrix, prevWeights: CPUMatrix, learningRate: number) {
        const weights = CPUMatrix.mul(error, CPUMatrix.transpose(prevWeights))
        for (const i in this.product.data) {
            const activation = this.activationFn.prime(this.product.data[i])
            this.error.data[i] = weights.data[i] * activation
        }
        const weightsDelta = CPUMatrix.mul(CPUMatrix.transpose(this.input), this.error)
        for (const i in weightsDelta.data) {
            this.weights.data[i] -= weightsDelta.data[i] * learningRate
        }
        for (const i in this.error.data) {
            this.biases.data[i] -= this.error.data[i] * learningRate
        }
        return this.error
    }
}