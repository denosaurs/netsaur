/**
 * This example shows how to use convolutional layers to extract features from an image.
 */

import {
  AveragePool2DLayer,
  Conv2DLayer,
  Cost,
  WASM,
  MaxPool2DLayer,
  Rank,
  Sequential,
  setupBackend,
  Tensor,
  tensor4D,
} from "../../mod.ts";
import { decode } from "https://deno.land/x/pngs@0.1.1/mod.ts";
import { createCanvas } from "https://deno.land/x/canvas@v1.4.1/mod.ts";
import { Layer } from "../../src/core/api/layer.ts";

const canvas = createCanvas(600, 600);
const ctx = canvas.getContext("2d");
ctx.fillStyle = "white";
ctx.fillRect(0, 0, 600, 600);

/**
 * Size of the input tensor for the neural network.
 */
const dim = 28;

/**
 * 3D array used as a filter for convolutional layers in a neural network.
 * The filter is a 3x3 matrix with values that can be used to detect vertical edges in an image.
 * The values in the filter are set such that the center column has negative values and the left and right columns have positive values.
 * This means that when the filter is convolved with an image, it will highlight vertical edges in the image.
 */
const kernel = [
  [
    [
      [0, -1, 1],
      [0, -1, 1],
      [0, -1, 1],
    ],
  ],
];

//Credit: Hashrock (https://github.com/hashrock)
const img = decode(Deno.readFileSync("./examples/filters/deno.png")).image;
const buffer = new Float32Array(dim * dim);
for (let i = 0; i < dim * dim; i++) {
  buffer[i] = img[i * 4];
}

/**
 * Setup the WASM backend. This backend is slower than the CPU backend but works on the Edge.
 */
await setupBackend(WASM);

/**
 * Draw the original image on the canvas.
 */
drawPixels(buffer, dim);

const conv = await feedForward([
  /**
   * Creates a convolutional layer in a neural network.
   */
  Conv2DLayer({
    /**
     * The kernel is set to the filter defined previously.
     */
    kernel: tensor4D(kernel),

    /**
     * Sets the kernel size for a convolutional layer in a neural network.
     * The kernel size is a 4D array that specifies the dimensions of the kernel that will be used to convolve the input tensor.
     * In this case, the kernel size is set to [1, 1, 3, 3],
     * which means that the kernel will have a depth of 1, a height of 3, and a width of 3.
     * This kernel size is used in the creation of two different convolutional layers in the code below the selection.
     */
    kernelSize: [1, 1, 3, 3],

    /**
     * Sets the padding for a convolutional layer in a neural network.
     * Padding is used to add extra pixels around the edges of an image
     * to ensure that the output of the convolution operation has the same dimensions as the input.
     * In this case, the padding is set to [1, 1],
     * which means that one pixel of padding will be added to the top, bottom, left, and right edges of the input tensor.
     */
    padding: [1, 1],
  }),
]);

drawPixels(conv.data, conv.shape[2], 280);

/**
 * Creates a network with a convolutional layer and a max pooling layer.
 */
const pool = await feedForward([
  Conv2DLayer({
    /**
     * The kernel is set to the filter defined previously.
     */
    kernel: tensor4D(kernel),

    /**
     * Sets the kernel size for a convolutional layer in a neural network.
     * The kernel size is a 4D array that specifies the dimensions of the kernel that will be used to convolve the input tensor.
     * In this case, the kernel size is set to [1, 1, 3, 3],
     * which means that the kernel will have a depth of 1, a height of 3, and a width of 3.
     * This kernel size is used in the creation of two different convolutional layers in the code below the selection.
     */
    kernelSize: [1, 1, 3, 3],

    /**
     * Sets the padding for a convolutional layer in a neural network.
     * Padding is used to add extra pixels around the edges of an image
     * to ensure that the output of the convolution operation has the same dimensions as the input.
     * In this case, the padding is set to [1, 1],
     * which means that one pixel of padding will be added to the top, bottom, left, and right edges of the input tensor.
     */
    padding: [1, 1],
  }),

  /**
   * Creates a Max Pooling layer in a neural network.
   * Max Pooling is a technique used in Convolutional Neural Networks (CNNs) to reduce the spatial dimensions of the output volume.
   * It is used to downsample the input along the spatial dimensions (width and height) while keeping the depth constant.
   * The Max Pooling layer takes a tensor as input and applies a max filter to non-overlapping subregions of the tensor.
   * The output of the Max Pooling layer is a tensor with reduced spatial dimensions and the same depth as the input tensor.
   */
  MaxPool2DLayer({
    /**
     * The strides specifies the stride of the pooling operation along the spatial dimensions.
     * In this case, the stride is set to [2, 2],
     * which means that the pooling operation will be applied to non-overlapping 2x2 regions of the input tensor.
     */
    strides: [2, 2],
  }),
]);

/**
 * Draw the output of the Max Pooling layer example on the canvas.
 */
drawPixels(pool.data, pool.shape[2], 0, 280, 2);

/**
 * Creates a network with a convolutional layer and an average pooling layer.
 */
const pool2 = await feedForward([
  Conv2DLayer({
    /**
     * The kernel is set to the filter defined previously.
     */
    kernel: tensor4D(kernel),

    /**
     * Sets the kernel size for a convolutional layer in a neural network.
     * The kernel size is a 4D array that specifies the dimensions of the kernel that will be used to convolve the input tensor.
     * In this case, the kernel size is set to [1, 1, 3, 3],
     * which means that the kernel will have a depth of 1, a height of 3, and a width of 3.
     * This kernel size is used in the creation of two different convolutional layers in the code below the selection.
     */
    kernelSize: [1, 1, 3, 3],

    /**
     * Sets the padding for a convolutional layer in a neural network.
     * Padding is used to add extra pixels around the edges of an image
     * to ensure that the output of the convolution operation has the same dimensions as the input.
     * In this case, the padding is set to [1, 1],
     * which means that one pixel of padding will be added to the top, bottom, left, and right edges of the input tensor.
     */
    padding: [1, 1],
  }),

  /**
   * Creates an average pooling layer in a neural network.
   * Average pooling is a technique used in Convolutional Neural Networks (CNNs) to reduce the spatial dimensions of the output volume.
   * It is used to downsample the input along the spatial dimensions (width and height) while keeping the depth constant.
   * The average pooling layer takes a tensor as input and applies an average filter to non-overlapping subregions of the tensor.
   * The output of the average pooling layer is a tensor with reduced spatial dimensions and the same depth as the input tensor.
   */
  AveragePool2DLayer({ strides: [2, 2] }),
]);

/**
 * Draw the output of the Average Pooling layer example on the canvas.
 */
drawPixels(pool2.data, pool2.shape[2], 280, 280, 2);

/**
 * Creates a network with the given layers and feeds forward the input data.
 */
async function feedForward(layers: Layer[]) {
  const net = new Sequential({
    size: [1, 1, dim, dim],
    silent: true,
    layers,
    cost: Cost.MSE,
  });

  const data = new Tensor(buffer, [1, 1, dim, dim]);
  return (await net.predict(data)) as Tensor<Rank.R4>;
}

/**
 * Function to draw the pixels on the canvas.
 */
function drawPixels(
  buffer: Float32Array,
  dim: number,
  offsetX = 0,
  offsetY = 0,
  scale = 1,
) {
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      const pixel = buffer[j * dim + i];
      ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
      ctx.fillRect(
        i * 10 * scale + offsetX,
        j * 10 * scale + offsetY,
        10 * scale,
        10 * scale,
      );
    }
  }
}

/**
 * Save the canvas as a PNG file.
 */
await Deno.writeFile("./examples/filters/output.png", canvas.toBuffer());
