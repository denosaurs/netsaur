import * as tf from '@tensorflow/tfjs-node';

const model = tf.sequential();
model.add(tf.layers.dense({inputShape:[2], units: 3, activation: 'sigmoid'}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

const xs = tf.tensor2d([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]);

const ys = tf.tensor2d([
    [0],
    [1],
    [1],
    [0],
]);

const start = Date.now();
model.fit(xs, ys, {epochs: 5000, verbose:0}).then(() => {
    console.log("Training took", Date.now() - start, "ms");
    model.predict(xs).print();
});
