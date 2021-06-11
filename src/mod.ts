import { Activation, LeakyRelu, Relu, Sigmoid, Tanh } from "./activation.ts";
import { Backend, CPUBackend, WebGPUBackend } from "./backend/backend.ts";
import { Matrix } from "./utils/matrix.ts";
export class NeuralNetwork {
  public inputSize: number;
  public hiddenSize: number;
  public outputSize: number;
  public activation: Activation;

  //could this be a vector
  public weightsHidden: Array<number>;
  public weightsOutput: Array<number>;

  public backend: any;

  constructor(
    inputSize: number,
    hiddenSize: number,
    outputSize: number,
    activation = "relu",
  ) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.activation = NeuralNetwork.activation(activation);
    this.weightsHidden = Array(hiddenSize).fill(Array(inputSize).fill(0)); // TODO initialization strategies, random noise etc...
    this.weightsOutput = Array(outputSize).fill(Array(hiddenSize).fill(0));
    this.backend = this.setupBackend();
  }

  async setupBackend(gpu = true): Promise<Backend> {
    if (!gpu) return new CPUBackend();

    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      console.log(`Found adapter: ${adapter.name}`);
      const features = [...adapter.features.values()];
      console.log(`Supported features: ${features.join(", ")}`);

      const device = await adapter.requestDevice();
      return new WebGPUBackend(adapter, device);
    } else {
      console.error("No adapter found");
      return new CPUBackend();
    }
  }

  static activation(activation: string): Activation {
    switch (activation) {
      case "sigmoid":
        return new Sigmoid();
      case "leaky-relu":
        return new LeakyRelu();
      case "tanh":
        return new Tanh();
      default:
        return new Relu();
    }
  }
  // TODO future: accept batch of inputs and comput batch output
  feed_forward(input: Array<number>): Array<number> {
    const hiddenLayerActivations = Matrix.dotMul(this.weightsHidden, input)
      .map(this.activation.activate);
    const outputActivations = Matrix.dotMul(
      this.weightsOutput,
      hiddenLayerActivations,
    )
      .map(this.activation.activate);
    return outputActivations;
  }

  backpropagate() {
  }

  train(
    inputs: Array<Array<number>>,
    _outputs: Array<Array<number>>,
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
