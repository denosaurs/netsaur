import { Tensor, type DataSet } from "../../packages/core/mod.ts";

export function assert(condition: boolean, message?: string) {
  if (!condition) {
    throw new Error(message);
  }
}

export function loadDataset(
  imagesFile: string,
  labelsFile: string,
  start: number,
  end: number,
  minibatch = 1,
) {
  const images = Deno.readFileSync(new URL(imagesFile, import.meta.url));
  const labels = Deno.readFileSync(new URL(labelsFile, import.meta.url));

  const imageView = new DataView(images.buffer);
  const labelView = new DataView(labels.buffer);

  assert(imageView.getUint32(0) === 0x803, "Invalid image file");
  assert(labelView.getUint32(0) === 0x801, "Invalid label file");

  const count = imageView.getUint32(4);
  assert(count === labelView.getUint32(4), "Image and label count mismatch");

  const inputs: Float32Array[] = [];
  let mean = 0;
  let sd = 0;
  for (let i = 0; i < count / minibatch; i++) {
    const input = new Float32Array(784 * minibatch);
    for (let j = 0; j < 784 * minibatch; j++) {
      input[j] = imageView.getUint8(16 + i * (784 * minibatch) + j);
      mean += input[j];
      sd += Math.pow(input[j], 2);
    }
    inputs.push(input);
  }

  mean /= count * 784;
  sd /= count * 784;
  sd -= Math.pow(mean, 2);
  sd = Math.sqrt(sd);

  const results: DataSet[] = [];

  for (let i = start; i < end / minibatch; i++) {
    for (let j = 0; j < 784 * minibatch; j++) {
      inputs[i][j] -= mean;
      inputs[i][j] /= sd;
    }

    const outputs = new Float32Array(10 * minibatch);
    for (let j = 0; j < minibatch; j++) {
      outputs[labelView.getUint8(8 + i * minibatch + j) + j * 10] = 1;
    }

    results.push({
      inputs: new Tensor(inputs[i], [minibatch, 1, 28, 28]),
      outputs: new Tensor(outputs, [minibatch, 10]),
    });
  }

  return results;
}
