const brain = require("brain.js");

const time = performance.now();

const config = {
  binaryThresh: 0.5,
  hiddenLayers: [4],
  activation: "sigmoid",
  leakyReluAlpha: 0.01,
};

const net = new brain.NeuralNetwork(config);

net.train([
  { input: [0, 0], output: [0] },
  { input: [1, 0], output: [1] },
  { input: [0, 1], output: [1] },
  { input: [1, 1], output: [0] },
], {
  iterations: 10000,
});

console.log(net.run([0, 0]));
console.log(net.run([1, 0]));
console.log(net.run([0, 1]));
console.log(net.run([1, 1]));
console.log(`time: ${performance.now() - time}ms`);
