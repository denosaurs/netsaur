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

    public feedForward(input: DataArray<T>, batches: number, type: DataType): DataArray<T> {
        const inputSize = this.input?.size || input.length / batches
        input = this.hidden[0].feedForward(input, batches, inputSize, type)

        for (let i = 1; i < this.hidden.length; i++) {
            const layer = this.hidden[i];
            const previousLayer = this.hidden[i - 1];
            input = layer.feedForward(input, batches, previousLayer.outputSize, type)
        }
        return input;
    }

    public backpropagate() {
    }

    public train(dataset: DataSet<T>, epochs: number, batches: number) {
        const type = this.input?.type || getType(dataset.inputs)

        for (let e = 0; e < epochs; e++) {
            this.feedForward(dataset.inputs, batches, type);

            // TODO loss function?

            this.backpropagate();
        }
    }

    public predict() {
    }
}
