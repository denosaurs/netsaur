import { ConvLayerConfig, LayerJSON } from "../core/types.ts";

/**
 * Convolutional layer.
 */
export class ConvLayer {
  type = "conv";
  load = false;
  data?: LayerJSON;
  constructor(public config: ConvLayerConfig) {}
  static fromJSON(layerJSON: LayerJSON): ConvLayer {
    if (layerJSON.type !== "conv") {
      throw new Error(
        "Cannot cannot create a Convolutional layer from a" +
          layerJSON.type.charAt(0).toUpperCase() + layerJSON.type.slice(1) +
          "Layer",
      );
    }
    if (layerJSON.padded === undefined || layerJSON.kernel === undefined) {
      throw new Error("Layer imported must be initialized");
    }
    const layer = new ConvLayer({
      kernel: layerJSON.kernel!.data,
      kernelSize: { x: layerJSON.kernel!.x, y: layerJSON.kernel!.y },
      padding: layerJSON.padding,
      strides: layerJSON.strides,
    });
    layer.load = true;
    layer.data = layerJSON;
    return layer;
  }
}
