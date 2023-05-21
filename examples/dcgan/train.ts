import {
  BatchNorm2DLayer,
  Conv2DLayer,
  ConvTranspose2DLayer,
  Cost,
  DenseLayer,
  Dropout2DLayer,
  FlattenLayer,
  LeakyReluLayer,
  Sequential,
} from "../../mod.ts";

function _makeGeneratorModel() {
  const model = new Sequential({
    size: [1, 100],
    layers: [
      DenseLayer({ size: [7 * 7 * 256] }),
      BatchNorm2DLayer(),
      LeakyReluLayer(),
      FlattenLayer({ size: [7, 7, 256] }),
      ConvTranspose2DLayer({ kernelSize: [128, 256, 5, 5] }),
      BatchNorm2DLayer(),
      LeakyReluLayer(),
      ConvTranspose2DLayer({ kernelSize: [64, 128, 5, 5], strides: [2, 2] }),
      BatchNorm2DLayer(),
      LeakyReluLayer(),
      ConvTranspose2DLayer({ kernelSize: [1, 64, 5, 5], strides: [2, 2] }),
    ],
    cost: Cost.CrossEntropy,
  });

  return model;
}

function _makeDiscriminatorModel() {
  const model = new Sequential({
    size: [1, 1, 28, 28],
    layers: [
      Conv2DLayer({ kernelSize: [64, 1, 5, 5], strides: [2, 2] }),
      LeakyReluLayer(),
      Dropout2DLayer({ probability: 0.3 }),
      Conv2DLayer({ kernelSize: [128, 64, 5, 5], strides: [2, 2] }),
      LeakyReluLayer(),
      Dropout2DLayer({ probability: 0.3 }),
      FlattenLayer({ size: [7 * 7 * 128] }),
      DenseLayer({ size: [1] }),
    ],
    cost: Cost.BinCrossEntropy,
  });

  return model;
}