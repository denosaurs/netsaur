import { DataArray, DataType } from "../../deps.ts";
import { DataSet, NetworkConfig, LayerConfig, Network, InputConfig, Cost } from "../types.ts";
import { getType } from "../util.ts";
import { CPUCostFunction, CrossEntropy, Hinge } from "./cost.ts";
import { CPULayer } from "./layer.ts";
import { CPUMatrix } from "./matrix.ts";

export class CPUNetwork<T extends DataType = DataType> implements Network {
    public input?: InputConfig;
    public hidden: CPULayer[];

    constructor(config: NetworkConfig) {
        this.input = config.input;
        this.hidden = config.hidden.map(layer => new CPULayer(layer));
    }

    public setCost(activation: Cost): void {
        let cost: CPUCostFunction;
        switch (activation) {
            case "crossentropy":
                cost = new CrossEntropy();
                break;
            case "hinge":
                cost = new Hinge();
                break;
        }
        this.hidden.map(layer => layer.costFunction = cost);
    }

    public addLayers(layers: LayerConfig[]): void {
        this.hidden.push(...layers.map(layer => new CPULayer(layer)));
    }

    public initialize(input: DataArray<T>, type: DataType, batches: number): DataArray<T> {
        const inputSize = this.input?.size ?? input.length / batches;
        this.hidden[0].initialize(type, inputSize, batches);

        for (let i = 1; i < this.hidden.length; i++) {
            const current = this.hidden[i];
            const previous = this.hidden[i - 1];
            current.initialize(type, previous.outputSize, batches);
        }
        return input;
    }

    public feedForward(input: CPUMatrix<T>): CPUMatrix<T> {
        for (const layer of this.hidden) {
            input = layer.feedForward(input);
        }
        return input;
    }
    
    public backpropagate() {
    }

    public train(dataset: DataSet<T>, epochs: number, batches: number): void {
        const type = this.input?.type || getType(dataset.inputs);

        for (let e = 0; e < epochs; e++) {
            this.initialize(dataset.inputs, type, batches);

            // TODO loss function?

            this.backpropagate();
        }
    }
    public get weights(): CPUMatrix<T>[] {
        return this.hidden.map(layer => layer.weights);
    }

    public predict() {
    }
}
