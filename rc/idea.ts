import { Activation, LeakyRelu, Relu, Sigmoid, Tanh } from "./activation.ts";
import { Backend, CPUBackend, WebGPUBackend } from "./backend/backend.ts";

export class NeuralNetwork {
    public inputSize: number;
    public hiddenSize: number[];
    public outputSize: number;

    public weightsHidden: Float32Array[];
    public weightsOutput: Float32Array;

    public activation: Activation;
    public backend: Backend;

    constructor(
        inputSize: number,
        hiddenSize: number[],
        outputSize: number,
        activation = "relu",
    ) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.activation = this.setupActivation(activation);
    }

    init() {
        this.weightsHidden = Array(hiddenSize).fill(Array(inputSize).fill(0)); // TODO initialization strategies, random noise etc...
        this.weightsOutput = Array(outputSize).fill(Array(hiddenSize).fill(0));
    }

    // TODO future: accept batch of inputs and compute batch output
    feed_forward(input: Float32Array): Array<number> {
        for (let i = 0; i < this.hiddenSize.length; i++) {
            
            
        }
        const hiddenLayerActivations = this.backend.matMul(this.weightsHidden, input)
            .map(this.activation.activate);
        const outputActivations = this.backend.matMul(
            this.weightsOutput,
            hiddenLayerActivations,
        ).map(this.activation.activate);
        return outputActivations;
    }

    backpropagate() {
    }

    train(
        inputs: Float32Array[],
        outputs: Float32Array[],
        epochs = 1000,
    ) {
        // TODO batch size for batch gradient descent
        for (let e = 0; e < epochs; e++) {
            for (let i = 0; i < inputs.length; i++) {
                this.feed_forward(inputs[i]);

                // TODO loss function?

                this.backpropagate();
            }
        }
    }

    predict() {
    }
}
