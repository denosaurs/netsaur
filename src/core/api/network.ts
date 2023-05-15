import { Backend, DataSet, NetworkConfig } from "../types.ts";
import { Tensor } from "../tensor/tensor.ts";
import { Rank } from "./shape.ts";

/**
 * Base Neural Network Structure. All Neural Networks should implement this.
 */
export interface NeuralNetwork {
  /**
   * The backend used by the Neural Network.
   */
  backend: Backend;

  /**
   * The configuration of the Neural Network.
   */
  config: NetworkConfig;

  /**
   * The train method is a function that trains a neural network using a set of training data.
   * It takes in an array of DataSet objects, the number of epochs to train for, and the learning rate.
   * The method modifies the weights and biases of the network to minimize the cost function and improve its accuracy on the training data.
   *
   * ```ts
   * network.train([
   *  { input: [0, 0], output: [0] },
   *  { input: [0, 1], output: [1] },
   *  { input: [1, 0], output: [1] },
   *  { input: [1, 1], output: [0] },
   * ]);
   * ```
   */
  train(datasets: DataSet[], epochs?: number, rate?: number): void;

  /**
   * The predict method is a function that takes in a Tensor object
   * representing the input to the neural network and returns a Promise that resolves to a Tensor object representing the output of the network.
   * This method is used to make predictions on new data after the network has been trained.
   *
   * ```ts
   * const prediction = await net.predict(tensor1D([0, 0]));
   * console.log(prediction.data[0]);
   * ```
   */
  predict(data: Tensor<Rank>): Promise<Tensor<Rank>>;

  /**
   * The save method saves the network to a Uint8Array.
   * This method is used to save the network after it has been trained.
   *
   * ```ts
   * const modelData = network.save();
   * Deno.writeFileSync("model.st", modelData);
   * ```
   */
  save(): Uint8Array;

  /**
   * The saveFile method takes in a string representing the path to a file to the safetensors format and saves the network to that file.
   * This method is used to save the network after it has been trained.
   *
   * ```ts
   * network.saveFile("model.st");
   * ```
   */
  saveFile(path: string): void;
}
