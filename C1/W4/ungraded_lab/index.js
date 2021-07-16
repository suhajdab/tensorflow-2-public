let mobilenet;
let model;
const webcam = new Webcam(document.getElementById("wc"));
const dataset = new RPSDataset();
var rockSamples = 0,
  paperSamples = 0,
  scissorsSamples = 0;
let isPredicting = false;

const getTrainableMobilenet = async () => {
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
  );
  const layer = mobilenet.getLayer("conv_pw_13_relu");
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
};

const onBatchEnd = async (_, logs) => {
  loss = logs.loss.toFixed(5);
  console.log({ loss });
};

const train = async () => {
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      tf.layers.dense({ units: 100, activation: "relu" }),
      tf.layers.dense({ units: 3, activation: "softmax" }),
    ],
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd,
    },
  });
};

const captureSample = (elem) => {
  switch (elem.id) {
    case "0":
      rockSamples++;
      document.getElementById("rocksamples").innerText =
        "Rock samples: " + rockSamples;
      break;
    case "1":
      paperSamples++;
      document.getElementById("papersamples").innerText =
        "Paper samples: " + paperSamples;
      break;
    case "2":
      scissorsSamples++;
      document.getElementById("scissorssamples").innerText =
        "Scissors samples: " + scissorsSamples;
      break;
  }
  label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);
};

const predict = async () => {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch (classId) {
      case 0:
        predictionText = "🪨";
        break;
      case 1:
        predictionText = "📃";
        break;
      case 2:
        predictionText = "✂️";
        break;
    }
    document.getElementById("prediction").innerText = predictionText;

    predictedClass.dispose();
    await tf.nextFrame();
  }
};

const doTraining = () => {
  train();
};

const startPredicting = () => {
  isPredicting = true;
  predict();
};

const stopPredicting = () => {
  isPredicting = false;
  predict();
};

(async () => {
  await webcam.setup();
  mobilenet = await getTrainableMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
})();
