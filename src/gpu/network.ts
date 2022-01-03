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

    public async feedForward(input: WebGPUData, type: DataType, batches: number) {
        const inputSize = this.input?.size || input.length / batches
        input = await this.hidden[0].feedForward(input, type, batches, inputSize)

        for (let i = 1; i < this.hidden.length; i++) {
            const layer = this.hidden[i];
            const previousLayer = this.hidden[i - 1];
            input = await layer.feedForward(input, type, batches, previousLayer.outputSize)
        }

        return input;
    }

    public backpropagate() {
    }

    public async train(datasets: DataSet<T>[], epochs: number, batches: number) {
        const type = getType(datasets[0].input)

        for (let e = 0; e < epochs; e++) {
            for (const dataset of datasets) {
                const input = await WebGPUData.from(this.backend, dataset.input);
                await this.feedForward(input, type, batches);

                // TODO loss function?

                this.backpropagate();
            }
        }
    }

    public predict() {
    }
}
