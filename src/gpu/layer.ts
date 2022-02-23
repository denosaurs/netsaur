import { WebGPUData, WebGPUBackend, DataType, DataArrayConstructor, DataArray } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
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
    public biasesDelta!: GPUMatrix

    private backend: WebGPUBackend

    constructor(config: GPULayerConfig, backend: WebGPUBackend) {
        this.outputSize = config.size
        this.setActivation(config.activation)
        this.backend = backend
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
    public async feedForward(
        input: GPUMatrix, 
        batches: number, 
        inputSize: number, 
        type: DataType
    ): Promise<WebGPUData> {
        const outputs = new WebGPUData(this.backend, type, this.outputSize * batches);
        if (!this.weights) await this.initialize(inputSize, type)

        await matMul(
            this.backend, 
            input, 
            this.weights, 
            outputs,
            this.activationFn.activate(type)
        )
        return outputs
    }
    
    public async initialize(inputSize: number, type: DataType) {
        const length = this.outputSize * inputSize
        const data = new DataArrayConstructor[type](length) as DataArray
        data.fill(1)
        this.weights = await WebGPUData.from(this.backend, data);
    }
}