export interface LossFunction {
    /** Return the cost associated with an output `a` and desired output `y`. */
    fn(a: number, y: number): number

    /** Return the error delta from the output layer. */
    delta(z: number, a: number, y: number): number
}

export class Quadratic implements LossFunction {
    public fn(a: number, y: number) {
        return 0.5 * np.linalg.norm(a - y) ** 2
    }

    public delta(z: number, a: number, y: number) {
        return (a - y) * this.sigmoidPrime(z)
    }

    /** Derivative of the sigmoid function. **/
    private sigmoidPrime(z: number) {
        return this.sigmoid(z) * (1 - this.sigmoid(z))
    }

    /** The sigmoid function. **/
    private sigmoid(val: number) {
        return 1 / (1 + Math.exp(-val))
    }
}


export class CrossEntropy implements LossFunction {
    public fn(a: number, y: number) {
        return np.sum(-y * Math.log(a) - (1 - y) * Math.log(1 - a))
    }

    public delta(z: number, a: number, y: number) {
        return a - y
    }
}
