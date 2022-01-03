import { WebGPUData, WebGPUBackend, DataType } from "../../deps.ts";
import { Activation, LayerConfig } from "../types.ts";
import { GPUActivationFn, Sigmoid, LeakyRelu, Tanh, Relu } from "./activation.ts";
import { matMul } from "./matmul.ts";

interface GPULayerConfig extends LayerConfig {
    size: number
    activation: Activation
}

export class GPULayer {
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

    public async feedForward(
        input: WebGPUData, 
        type: DataType, 
        batches: number,
        inputSize: number
    ): Promise<WebGPUData> {
        const outputs = new WebGPUData(this.backend, type, this.outputSize * batches);
        await matMul(
            this.backend, 
            input, 
            this.weights, 
            outputs, 
            batches,
            inputSize,
            this.outputSize,
            this.activationFn.activate(type)
        )
        return outputs
    }
}