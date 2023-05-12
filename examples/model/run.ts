import { CPU, Sequential, setupBackend, tensor2D } from "../../mod.ts";

/**
 * Setup the CPU backend. This backend is fast but doesn't work on the Edge.
 */
await setupBackend(CPU);

const model = Sequential.loadFile("examples/model/xor.test.st");

/**
 * Predict the output of the XOR function for the given inputs.
 */
const out1 = (await model.predict(tensor2D([[0, 0]]))).data;
console.log(`0 xor 0 = ${out1[0]} (should be close to 0)`);

const out2 = (await model.predict(tensor2D([[1, 0]]))).data;
console.log(`1 xor 0 = ${out2[0]} (should be close to 1)`);

const out3 = (await model.predict(tensor2D([[0, 1]]))).data;
console.log(`0 xor 1 = ${out3[0]} (should be close to 1)`);

const out4 = (await model.predict(tensor2D([[1, 1]]))).data;
console.log(`1 xor 1 = ${out4[0]} (should be close to 0)`);
