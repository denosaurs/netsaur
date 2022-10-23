import { PoolCPULayer } from "../backends/cpu/layers/pool.ts";
import { CPUMatrix } from "../backends/cpu/matrix.ts";
import { PoolLayer } from "../layers/mod.ts";

const pool = PoolLayer({ strides: 2 }) as PoolCPULayer;

const input = new CPUMatrix(new Float32Array([
    1, 0, 0, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 0, 0, 1,
]), 4, 4)
pool.initialize({x: 4, y: 4}, 1)
const output = pool.feedForward(input)!
console.log(input.fmt())
console.log(output.fmt())
const prevError = new CPUMatrix(new Float32Array([
    1, 2,
    3, 4,
]), 2, 2)
pool.backPropagate(prevError, 1)
const error = pool.getError()
console.log(error.fmt())
