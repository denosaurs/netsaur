const tf = require("@tensorflow/tfjs-node");

async function predictOutput() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 8, inputShape: 2, activation: "tanh" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: "sgd", loss: "meanSquaredError", lr: 0.6 });

  // Creating dataset
  const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const ys = tf.tensor2d([[0], [1], [1], [0]]);
  const time = performance.now();

  // Train the model
  await model.fit(xs, ys, {
    batchSize: 1,
    epochs: 10000,
    verbose: false
  });
  console.log(`training time: ${performance.now() - time}ms`);
}

predictOutput();
