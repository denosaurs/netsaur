import { DataArray, DataType } from "../../deps.ts";
import { DataSet, NetworkConfig, LayerConfig, Network, InputConfig } from "../types.ts";
import { getType } from "../util.ts";
import { CPULayer } from "./layer.ts";

export class CPUNetwork<T extends DataType = DataType> implements Network {
    public input?: InputConfig;
    public hidden: CPULayer<T>[];

    constructor(config: NetworkConfig) {
        this.input = config.input
        this.hidden = config.hidden.map(layer => new CPULayer(layer))
    }

    public addLayers(layers: LayerConfig[]) {
        this.hidden.push(...layers.map(layer => new CPULayer(layer)))
    }

    public feedForward(input: DataArray<T>, type: DataType, batches: number): DataArray<T> {
        const inputSize = this.input?.size || input.length / batches
        input = this.hidden[0].feedForward(input, type, batches, inputSize)

        for (let i = 1; i < this.hidden.length; i++) {
            const layer = this.hidden[i];
            const previousLayer = this.hidden[i - 1];
            input = layer.feedForward(input, type, batches, previousLayer.outputSize)
        }
        return input;
    }

    public backpropagate() {
    }

    public train(datasets: DataSet<T>[], epochs: number, batches: number) {
        const type = getType(datasets[0].input)

        for (let e = 0; e < epochs; e++) {
            for (const dataset of datasets) {
                this.feedForward(dataset.input, type, batches);

                // TODO loss function?

                this.backpropagate();
            }
        }
    }

    public predict() {
    }
}
