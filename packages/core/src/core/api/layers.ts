import { Activation, LayerType } from "../types.ts";
import {
  type BatchNormLayerConfig,
  type Conv2DLayerConfig,
  type ConvTranspose2DLayerConfig,
  type DenseLayerConfig,
  type DropoutLayerConfig,
  type EmbeddingLayerConfig,
  type Conv1DLayerConfig,
  type ConvTranspose1DLayerConfig,
  type LSTMLayerConfig,
  type Layer,
  type Pool2DLayerConfig,
  PoolMode,
  type SoftmaxLayerConfig,
} from "./layer.ts";

/**
 * Creates a dense layer (also known as a fully connected layer).
 * Dense layers feed all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer.
 * See https://en.wikipedia.org/wiki/Feedforward_neural_network#Fully_connected_network
 */
export function DenseLayer(config: DenseLayerConfig): Layer {
  return { type: LayerType.Dense, config };
}

/**
 * Creates a dropout layer. Dropout is a regularization technique for reducing overfitting.
 * The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections.
 *  See https://en.wikipedia.org/wiki/Dropout_(neural_networks)
 */
export function Dropout1DLayer(config: DropoutLayerConfig): Layer {
  return { type: LayerType.Dropout1D, config };
}

/**
 * Creates a dropout layer. Dropout is a regularization technique for reducing overfitting.
 * The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections.
 *  See https://en.wikipedia.org/wiki/Dropout_(neural_networks)
 */
export function Dropout2DLayer(config: DropoutLayerConfig): Layer {
  return { type: LayerType.Dropout2D, config };
}

/**
 * Creates a convolutional layer.
 * Convolutional layers are used for feature extraction.
 * They are commonly used in image processing.
 * See https://en.wikipedia.org/wiki/Convolutional_neural_network
 */
export function Conv2DLayer(config: Conv2DLayerConfig): Layer {
  return { type: LayerType.Conv2D, config };
}

/**
 * Creates a 1D convolutional layer.
 * Convolutional layers are used for feature extraction.
 * They are commonly used in image processing.
 * See https://en.wikipedia.org/wiki/Convolutional_neural_network
 */
export function Conv1DLayer(config: Conv1DLayerConfig): Layer {
  return { type: LayerType.Conv1D, config };
}

export function ConvTranspose2DLayer(
  config: ConvTranspose2DLayerConfig,
): Layer {
  return { type: LayerType.ConvTranspose2D, config };
}

export function ConvTranspose1DLayer(
  config: ConvTranspose1DLayerConfig,
): Layer {
  return { type: LayerType.ConvTranspose1D, config };
}

/**
 * Creates a pooling layer.
 * Pooling layers are used for downsampling.
 * See https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
 */
export function Pool2DLayer(config: Pool2DLayerConfig): Layer {
  return { type: LayerType.Pool2D, config };
}

/**
 * Creates a max pooling layer.
 * Pooling layers are used for downsampling.
 * See https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
 */
export function MaxPool2DLayer(config: Pool2DLayerConfig): Layer {
  config.mode = PoolMode.Max;
  return { type: LayerType.Pool2D, config };
}

/**
 * Creates an average pooling layer.
 * Pooling layers are used for downsampling.
 * See https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
 */
export function AveragePool2DLayer(config: Pool2DLayerConfig): Layer {
  config.mode = PoolMode.Avg;
  return { type: LayerType.Pool2D, config };
}

/**
 * Creates a softmax layer. Softmax layers are used for classification.
 * See https://en.wikipedia.org/wiki/Softmax_function
 */
export function SoftmaxLayer(config: SoftmaxLayerConfig = {}): Layer {
  return { type: LayerType.Softmax, config };
}

/**
 * Creates a sigmoid layer. Sigmoid layers use the sigmoid activation function.
 * See https://en.wikipedia.org/wiki/Sigmoid_function
 */
export function SigmoidLayer(): Layer {
  const config = { activation: Activation.Sigmoid };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a leaky relu layer.
 * Leaky relu layers use the leaky relu activation function.
 */
export function LeakyReluLayer(): Layer {
  const config = { activation: Activation.LeakyRelu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a tanh layer.
 * Tanh layers use the tanh activation function.
 */
export function TanhLayer(): Layer {
  const config = { activation: Activation.Tanh };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a relu layer.
 * Relu layers use the relu activation function.
 */
export function ReluLayer(): Layer {
  const config = { activation: Activation.Relu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a relu6 layer.
 * Relu6 layers use the relu6 activation function.
 */
export function Relu6Layer(): Layer {
  const config = { activation: Activation.Relu6 };
  return { type: LayerType.Activation, config };
}

/**
 * Creates an Elu layer.
 * Elu layers use the elu activation function.
 */
export function EluLayer(): Layer {
  const config = { activation: Activation.Elu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a Selu layer.
 * Selu layers use the selu activation function.
 */
export function SeluLayer(): Layer {
  const config = { activation: Activation.Selu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates a Gelu layer.
 * Gelu layers use the Gelu activation function.
 */
export function GeluLayer(): Layer {
  const config = { activation: Activation.Gelu };
  return { type: LayerType.Activation, config };
}

/**
 * Creates an Embedding layer.
 * Embedding layers convert numerical data into dense vectors.
 * They are usually used for text processing.
 */
export function EmbeddingLayer(config: EmbeddingLayerConfig): Layer {
  return { type: LayerType.Embedding, config };
}

/**
 * Creates a Flatten layer.
 * Flatten layers flatten the input.
 * They are usually used to transition from convolutional layers to dense layers.
 */
export function FlattenLayer(): Layer {
  return { type: LayerType.Flatten };
}

/**
 * Creates a Flatten layer.
 * Flatten layers flatten the input.
 * They are usually used to transition from convolutional layers to dense layers.
 */
export function LSTMLayer(config: LSTMLayerConfig): Layer {
  return { type: LayerType.LSTM, config };
}

/**
 * Creates a BatchNorm1D layer.
 * BatchNorm1D layers normalize the input.
 * They are usually used to improve the speed, performance, and stability of neural networks.
 * See https://en.wikipedia.org/wiki/Batch_normalization
 */
export function BatchNorm1DLayer(config: BatchNormLayerConfig = {}): Layer {
  config.epsilon = config.epsilon || 0.001;
  config.momentum = config.momentum || 0.99;
  return { type: LayerType.BatchNorm1D, config };
}

/**
 * Creates a BatchNorm2D layer.
 * BatchNorm2D layers normalize the input.
 * They are usually used to improve the speed, performance, and stability of neural networks.
 * See https://en.wikipedia.org/wiki/Batch_normalization
 */
export function BatchNorm2DLayer(config: BatchNormLayerConfig = {}): Layer {
  config.epsilon = config.epsilon || 0.001;
  config.momentum = config.momentum || 0.99;
  return { type: LayerType.BatchNorm2D, config };
}
