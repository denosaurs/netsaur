import { DataArray, DataType } from "../../deps.ts";
import { DataSet, NetworkConfig, LayerConfig, Network, InputConfig, Cost } from "../types.ts";
import { getType } from "../util.ts";
import { CPUCostFunction, CrossEntropy, Hinge } from "./cost.ts";
import { CPULayer } from "./layer.ts";
import { CPUMatrix } from "./matrix.ts";

export class CPUNetwork<T extends DataType = DataType> implements Network {
    public input?: InputConfig;
    public hidden: CPULayer[];
    public output: CPULayer;

    constructor(config: NetworkConfig) {
        this.input = config.input;
        this.hidden = config.hidden.map(layer => new CPULayer(layer));
        this.output = new CPULayer(config.output)
    }

    public setCost(activation: Cost): void {
        let costFn: CPUCostFunction;
        switch (activation) {
            case "crossentropy":
                costFn = new CrossEntropy();
                break;
            case "hinge":
                costFn = new Hinge();
                break;
        }
        this.hidden.map(layer => layer.costFn = costFn);
    }

    public addLayers(layers: LayerConfig[]): void {
        this.hidden.push(...layers.map(layer => new CPULayer(layer)));
    }

    public initialize(type: DataType, inputSize: number, batches: number) {
        this.hidden[0].initialize(type, inputSize, batches);

        for (let i = 1; i < this.hidden.length; i++) {
            const current = this.hidden[i];
            const previous = this.hidden[i - 1];
            current.initialize(type, previous.outputSize, batches);
        }

        const lastLayer = this.hidden[this.hidden.length - 1]
        this.output.initialize(type, lastLayer.outputSize, batches);
    }

    public feedForward(input: CPUMatrix<T>): CPUMatrix<T> {
        for (const layer of this.hidden) {
            input = layer.feedForward(input);
        }
        input = this.output.feedForward(input)
        return input;
    }

    public backpropagate(output: DataArray<T>, learningRate: number) {
        for (const i in this.output.output.data) {
            const activation = this.output.activationFn.prime(this.output.product.data[i])
            const delta = this.output.costFn.prime(this.output.output.data[i], output[i]);
            this.output.error.data[i] = activation * delta
        }
        const inputs = CPUMatrix.transpose(this.output.input)
        const weightsDelta = CPUMatrix.mul(inputs, this.output.error)
        for (const i in weightsDelta.data) {
            this.output.weights.data[i] -= weightsDelta.data[i] * learningRate
        }
        for (const i in this.output.error.data) {
            this.output.biases.data[i] -= this.output.error.data[i] * learningRate
        }
        let error = this.output.error
        const lastLayer = this.hidden[this.hidden.length - 1]
        error = lastLayer.backPropagate(error, this.output.weights, learningRate)
        for (let i = this.hidden.length - 2; i >= 0; i--) {
            const prevLayer = this.hidden[i + 1];
            this.hidden[i].backPropagate(error, prevLayer.weights, learningRate)
        }
    }

    public train(dataset: DataSet<T>, epochs: number, batches: number, learningRate: number): void {
        const type = this.input?.type || getType(dataset.inputs);
        const inputSize = this.input?.size || dataset.inputs.length / batches;

        const input = new CPUMatrix(dataset.inputs, inputSize, batches, type)
        this.initialize(type, inputSize, batches);

        for (let e = 0; e < epochs; e++) {
            // TODO: do something with this output
            this.feedForward(input);

            this.backpropagate(dataset.outputs, learningRate);
        }
    }

    public get weights(): CPUMatrix<T>[] {
        return this.hidden.map(layer => layer.weights);
    }

    public getOutput(): DataArray<T> {
        return this.output.output.data as DataArray<T>
    }

    public predict() {
    }
}
