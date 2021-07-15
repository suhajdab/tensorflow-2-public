const fn = (x) => 2 * x - 3;
const getSeries = (n) => {
  const xs = Array.from(Array(n).keys());
  const ys = xs.map(fn);
  return [xs, ys, n];
};

const onEpochEnd = async (epoch, logs) => {
  if (epoch % 10 !== 0) return;
  console.log("Epoch:" + epoch + " Loss:" + logs.loss);
};

const doTraining = async (model) => {
  const history = await model.fit(xs, ys, {
    epochs: 500,
    callbacks: {
      onEpochEnd,
    },
  });
};

const doPrediction = () => {
  const xTest = 5;
  const predTensor = model.predict(tf.tensor2d([xTest], [1, 1]));
  const pred = Array.from(predTensor.dataSync());
  const real = fn(xTest);
  console.log(
    `%c prediction = ${pred[0]}, expected = ${real}`,
    "color: #f60; font-size: 12px"
  );
};

const [xSeries, ySeries, l] = getSeries(10);
const xs = tf.tensor2d(xSeries, [l, 1]);
const ys = tf.tensor2d(ySeries, [l, 1]);

console.log("series", { xSeries, ySeries });

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
model.summary();

doTraining(model).then(doPrediction);
