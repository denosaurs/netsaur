// @deno-types=./objc.d.ts
import { classes, objc } from "../../../../objc/index.js";

objc.import("MetalPerformanceShaders");
objc.import("Metal");

const {} = classes;

const device = MTLCreateSystemDefaultDevice();
console.log(device);

const queue = device.newCommandQueue();
console.log(queue);

// perfect, another bug in deno node-api implementation
// i love those
// seems like i gtg now.. will continue later
// you can try playing around MPS API using the deno_objc module. C functions aren't supported on there yet but you can dlopen them manually
// from the framework shared library (ex. MTLCreateSystemDefaultDevice is in dlopen(/System/Library/Frameworks/Metal.framework/Metal)).
// See the example MPS neural network code for reference on how to use the MPS API.
// can download here: https://developer.apple.com/documentation/metalperformanceshaders/training_a_neural_network_with_metal_performance_shaders
