import { ConvLayer, DenseLayer, NeuralNetwork } from "../mod.ts";
import { ConvCPULayer } from "../src/cpu/layers/conv.ts";
import { CPUMatrix } from "../src/cpu/matrix.ts";
import { CPUNetwork } from "../src/cpu/network.ts";

const kernel = new Float32Array([
  1, 0, 1,
  1, 0, 1,
  1, 0, 1,
])

const net = await new NeuralNetwork({
  silent: true,
  layers: [
    new ConvLayer({ 
      size: { x: 5, y: 5 }, 
      activation: "sigmoid", 
      kernel: kernel,
      kernelSize: {x: 3, y: 3},
      padding: 2
     }),
    new DenseLayer({ size: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
  input: 2,
}).setupBackend("cpu");

const buf = new Float32Array([
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
])
const input = new CPUMatrix(buf, 5, 5);
(net.network as CPUNetwork).initialize({ x: 5, y: 5 }, 1);
(net.network as CPUNetwork).layers[0].feedForward(input);
console.log(((net.network as CPUNetwork).layers[0] as ConvCPULayer).test.fmt())
console.log(((net.network as CPUNetwork).layers[0] as ConvCPULayer).output.fmt())
