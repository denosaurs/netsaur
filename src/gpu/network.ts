import { GPULayer } from "./layer.ts";
import { DataSet, InputConfig, LayerConfig, Network, NetworkConfig } from "../types.ts";
import { DataType, WebGPUBackend, WebGPUData } from "../../deps.ts";
import { getType } from "../util.ts";

export class GPUNetwork<T extends DataType = DataType> implements Network {
    public input?: InputConfig;
    public hidden: GPULayer[];
    public backend: WebGPUBackend;

    constructor(config: NetworkConfig, backend: WebGPUBackend) {
        this.input = config.input
        this.backend = backend
        this.hidden = config.hidden.map(layer => new GPULayer(layer, backend))
    }

    public addLayers(layers: LayerConfig[]) {
        this.hidden.push(...layers.map(layer => new GPULayer(layer, this.backend)))
    }

    public async feedForward(input: WebGPUData, batches: number, type: DataType) {
        const inputSize = this.input?.size || input.length / batches
        input = await this.hidden[0].feedForward(input, batches, inputSize, type)

        for (let i = 1; i < this.hidden.length; i++) {
            const layer = this.hidden[i];
            const previousLayer = this.hidden[i - 1];
            input = await layer.feedForward(input, batches, previousLayer.outputSize, type)
        }

        return input;
    }

    public backpropagate() {
    }

    public async train(dataset: DataSet<T>, epochs: number, batches: number) {
        const type = this.input?.type || getType(dataset.inputs)

        for (let e = 0; e < epochs; e++) {
            const input = await WebGPUData.from(this.backend, dataset.inputs);
            await this.feedForward(input, batches, type);

            // TODO loss function?

            this.backpropagate();
        }
    }

    public predict() {
    }
}
