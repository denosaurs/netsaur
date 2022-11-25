import { Rank, Shape } from "./types.ts"

export class MismatchedRankError extends Error {
    constructor(rank: number, expected: number) {
        super(`Tensor of rank ${rank} is not assignable to rank ${expected}`)
    }
}

export class InvalidFlattenError extends Error {
    constructor(input: Shape[Rank], output: Shape[Rank]) {
        super(`Cannot flatten tensor of shape ${input} to shape ${output}`)
    }
}