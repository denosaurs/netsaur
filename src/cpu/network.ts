import { DataType } from "../../deps.ts";
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

    public initialize(type: DataType, inputSize: number, batches: number){
        this.hidden[0].initialize(type, inputSize, batches);

        for (let i = 1; i < this.hidden.length; i++) {
            const current = this.hidden[i];
            const previous = this.hidden[i - 1];
            current.initialize(type, previous.outputSize, batches);
        }
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
        const inputSize = this.input?.size || dataset.inputs.length / batches;

        const input = new CPUMatrix(dataset.inputs, inputSize, batches, type)

        for (let e = 0; e < epochs; e++) {
            this.initialize(type, inputSize, batches);

            this.feedForward(input);

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
