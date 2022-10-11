import type { Dataset } from "../../backends/native/mod.ts";
import { Matrix } from "../../backends/native/mod.ts";

export function assert(condition: boolean, message?: string) {
  if (!condition) {
    throw new Error(message);
  }
}

export function loadDataset(imagesFile: string, labelsFile: string) {
  const images = Deno.readFileSync(new URL(imagesFile, import.meta.url));
  const labels = Deno.readFileSync(new URL(labelsFile, import.meta.url));

  const imageView = new DataView(images.buffer);
  const labelView = new DataView(labels.buffer);

  assert(imageView.getUint32(0) === 0x803, "Invalid image file");
  assert(labelView.getUint32(0) === 0x801, "Invalid label file");

  const count = imageView.getUint32(4);
  assert(count === labelView.getUint32(4), "Image and label count mismatch");

  const results: Dataset[] = [];

  for (let i = 0; i < count; i++) {
    const inputs = new Float32Array(784);
    for (let j = 0; j < 784; j++) {
      inputs[j] = imageView.getUint8(16 + i * 784 + j) / 255;
    }

    const outputs = new Float32Array(10);
    outputs[labelView.getUint8(8 + i)] = 1;

    results.push({
      inputs: new Matrix(1, inputs.length, inputs),
      outputs: new Matrix(1, outputs.length, outputs),
    });
  }

  return results;
}
