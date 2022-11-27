import { Rank, Shape } from "./types.ts"

export class IncompatibleRankError extends Error {
    constructor(shape: number, expected: number) {
        super(`Layer of rank ${expected} is incompatible with tensor of rank ${shape}`)
    }
}

export class InvalidFlattenError extends Error {
    constructor(input: Shape[Rank], output: Shape[Rank]) {
        super(`Cannot flatten tensor of shape ${input} to shape ${output}`)
    }
}

export class NoWebGPUBackendError extends Error {
    constructor() {
        super(`WebGPU backend not initialized. Help: Did you forget to call setupBackend()?`)
    }
}