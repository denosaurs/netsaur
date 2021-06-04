import { Activation, Relu, LeakyRelu, Tanh, Sigmoid } from './activation.ts'
import { Matrix, MatrixGPU, Backend } from './utils/mod.ts'

export class NeuralNetwork {

    public inputSize: number;
    public hiddenSize: number;
    public outputSize: number;
    public activation: Activation;

    //could this be a vector
    public weightsHidden: Array<number>;
    public weightsOutput: Array<number>;

    public backend: any;

    constructor(inputSize: number, hiddenSize: number, outputSize: number, activation = 'relu') {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.activation = this.setActivation(activation);
        this.weightsHidden = Array(hiddenSize).fill(Array(inputSize).fill(0)); // TODO initialization strategies, random noise etc...
        this.weightsOutput = Array(outputSize).fill(Array(hiddenSize).fill(0));
    }
    
    async setupBackend(gpu = true) {
        if (!gpu) return new MatrixCPU();

        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
            console.log(`Found adapter: ${adapter.name}`);
            const features = [...adapter.features.values()];
            console.log(`Supported features: ${features.join(", ")}`);

            const device = await adapter.requestDevice()
            return new MatrixGPU(adapter, device)
        } else {
            console.error("No adapter found");
            return new MatrixCPU()
        }
    }
    
    setActivation(activation:string): Activation {
        switch(activation) {
            case 'sigmoid':
                return new Sigmoid();
            case 'leaky-relu':
                return new LeakyRelu();
            case 'tanh':
                return new Tanh();
            default:
                return new Relu();
        }
    }
    // TODO future: accept batch of inputs and comput batch output
    feed_forward(input: Array<number>): Array<number> {
        const hiddenLayerActivations = Matrix.dotMul(this.weightsHidden, input)
            .map(this.activation.activate);
        const outputActivations = Matrix.dotMul(this.weightsOutput, hiddenLayerActivations)
            .map(this.activation.activate);
        return outputActivations
    }

    backpropagate() {

    }

    train(inputs: Array<Array<number>>, outputs: Array<Array<number>>, epochs = 1000) {
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