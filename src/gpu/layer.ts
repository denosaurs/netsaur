import { WebGPUData, WebGPUBackend, DataType, DataArrayConstructor, DataArray } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { GPUActivationFn, Sigmoid, LeakyRelu, Tanh, Relu } from "./activation.ts";
import { matMul } from "./kernels/matmul.ts";

interface GPULayerConfig extends LayerConfig {
    size: number
    activation: Activation
}

export class GPULayer<T extends DataType = DataType> {
    public outputSize: number
    public weights!: WebGPUData
    public activationFn: GPUActivationFn = new Sigmoid()

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

    public async feedForward(
        input: WebGPUData, 
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
            inputSize,
            this.outputSize,
            batches,
            this.activationFn.activate(type)
        )
        return outputs
    }
    
    public async initialize(inputSize: number, type: DataType) {
        const length = this.outputSize * inputSize
        const data = new DataArrayConstructor[type](length) as DataArray<T>
        data.fill(1)
        this.weights = await WebGPUData.from(this.backend, data);
    }
}