import { Activation, DenseLayerConfig, LayerJSON } from "../core/types.ts";

/**
 * Base class for all layers.
 */
export class DenseLayer {
  type = "dense";
  load = false;
  data?: LayerJSON;
  constructor(public config: DenseLayerConfig) {}
  static fromJSON(layerJSON: LayerJSON): DenseLayer {
    if (layerJSON.type !== "dense") {
      throw new Error(
        "Cannot cannot create a Dense layer from a" +
          layerJSON.type.charAt(0).toUpperCase() + layerJSON.type.slice(1) +
          "Layer",
      );
    }
    if (layerJSON.weights === undefined || layerJSON.biases === undefined) {
      throw new Error("Layer imported must be initialized");
    }
    const layer = new DenseLayer({
      size: layerJSON.outputSize,
      activation: (layerJSON.activationFn as Activation) || "sigmoid",
    });
    layer.load = true;
    layer.data = layerJSON;
    return layer;
  }
}
