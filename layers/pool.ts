import { LayerJSON, PoolLayerConfig } from "../core/types.ts";

/**
 * MaxPool layer.
 */
export class PoolLayer {
  type = "pool";
  load = false;
  data?: LayerJSON;
  constructor(public config: PoolLayerConfig) {}
  static fromJSON(layerJSON: LayerJSON): PoolLayer {
    if (layerJSON.type !== "pool") {
      throw new Error(
        "Cannot cannot create a MaxPooling layer from a" +
          layerJSON.type.charAt(0).toUpperCase() + layerJSON.type.slice(1) +
          "Layer",
      );
    }
    if (layerJSON.strides === undefined) {
      throw new Error("Layer imported must be initialized");
    }
    const layer = new PoolLayer({ strides: layerJSON.strides! });
    layer.load = true;
    layer.data = layerJSON;
    return layer;
  }
}
