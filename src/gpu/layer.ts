import { WebGPUBackend, DataType } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { fromType } from "../util.ts";
import { GPUActivationFn, Sigmoid, LeakyRelu, Tanh, Relu } from "./activation.ts";
import { CrossEntropy, GPUCostFunction } from "./cost.ts";
import { matMul } from "./kernels/matmul.ts";
import { GPUMatrix } from "./matrix.ts";

interface GPULayerConfig extends LayerConfig {
    size: number
    activation: Activation
}

export class GPULayer<T extends DataType = DataType> {
    public outputSize: number
    public activationFn: GPUActivationFn = new Sigmoid()
    public costFunction: GPUCostFunction = new CrossEntropy()

    public weights!: GPUMatrix
    public product!: GPUMatrix
    public output!: GPUMatrix
    public weightsDelta!: GPUMatrix
    public error!: GPUMatrix

    private backend: WebGPUBackend

    constructor(config: GPULayerConfig, backend: WebGPUBackend) {
        this.outputSize = config.size
        this.setActivation(config.activation)
        this.backend = backend
    }

    public async initialize(type: DataType, inputSize: number, batches: number) {
        const data = new (fromType(type))(this.outputSize * inputSize).fill(1);

        this.weights = await GPUMatrix.from(this.backend, data, this.outputSize, inputSize, type)
        this.output = await GPUMatrix.with(this.backend, this.outputSize, batches, type)
        this.product = await GPUMatrix.with(this.backend, this.outputSize, batches, type)
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

    // TODO: memoization
    public async feedForward(input: GPUMatrix): Promise<GPUMatrix> {
        await matMul(
            this.backend, 
            input, 
            this.weights, 
            this.output,
            this.activationFn.activate(input.type)
        )
        return this.output
    }
}