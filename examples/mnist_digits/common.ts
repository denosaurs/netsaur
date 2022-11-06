import { DataSet } from "../../core/types.ts";
import { Tensor } from "../../mod.ts";

export function assert(condition: boolean, message?: string) {
  if (!condition) {
    throw new Error(message);
  }
}

export function loadDataset(imagesFile: string, labelsFile: string, start: number, end: number) {
  const images = Deno.readFileSync(new URL(imagesFile, import.meta.url));
  const labels = Deno.readFileSync(new URL(labelsFile, import.meta.url));

  const imageView = new DataView(images.buffer);
  const labelView = new DataView(labels.buffer);

  assert(imageView.getUint32(0) === 0x803, "Invalid image file");
  assert(labelView.getUint32(0) === 0x801, "Invalid label file");

  const count = imageView.getUint32(4);
  assert(count === labelView.getUint32(4), "Image and label count mismatch");

  const inputs: Float32Array[] = [];
  let mean = 0
  let sd = 0
  for (let i = 0; i < count; i++) {
    const input = new Float32Array(784);
    for (let j = 0; j < 784; j++) {
      input[j] = imageView.getUint8(16 + i * 784 + j);
      mean += input[j]
      sd += Math.pow(input[j], 2)
    }
    inputs.push(input)
  }

  mean /= count * 784
  sd /= count * 784
  sd -= Math.pow(mean, 2)
  sd = Math.sqrt(sd)

  const results: DataSet[] = [];
  for (let i = start; i < end; i++) {
    for (let j = 0; j < 784; j++) {
      inputs[i][j] -= mean;
      inputs[i][j] /= sd;
    }

    const outputs = new Float32Array(10);
    outputs[labelView.getUint8(8 + i)] = 1;

    results.push({
      inputs: new Tensor(inputs[i], [28, 28, 1, 1]),
      outputs: new Tensor(outputs, [10]),
    });
  }

  return results;
}
