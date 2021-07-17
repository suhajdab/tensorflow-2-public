let mobilenet;
let model;
const webcam = new Webcam(document.getElementById("wc"));
const dataset = new RPSDataset();
var rockSamples = 0,
  paperSamples = 0,
  scissorsSamples = 0,
  spockSamples = 0,
  lizardSamples = 0;
let isPredicting = false;

const loadMobilenet = async () => {
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json"
  );
  const layer = mobilenet.getLayer("conv_pw_13_relu");
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
};

const onBatchEnd = async (_, logs) => {
  console.log({ ...logs });
};

const train = async () => {
  dataset.ys = null;
  dataset.encodeLabels(5);

  // In the space below create a neural network that can classify hand gestures
  // corresponding to rock, paper, scissors, lizard, and spock. The first layer
  // of your network should be a flatten layer that takes as input the output
  // from the pre-trained MobileNet model. Since we have 5 classes, your output
  // layer should have 5 units and a softmax activation function. You are free
  // to use as many hidden layers and neurons as you like.
  // HINT: Take a look at the Rock-Paper-Scissors example. We also suggest
  // using ReLu activation functions where applicable.
  model = tf.sequential({
    layers: [
      // YOUR CODE HERE
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      tf.layers.dropout({ rate: 0.25 }),
      tf.layers.dense({ units: 128, activation: "relu" }),
      tf.layers.dense({ units: 5, activation: "softmax" }),
    ],
  });

  model.summary();

  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  const optimizer = tf.train.adam(0.0001); // YOUR CODE HERE
  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 50,
    callbacks: {
      onBatchEnd,
    },
  });
};

const addSample = (id) => {
  switch (id) {
    case 0:
      document.getElementById("rocksamples").value = ++rockSamples;
      break;
    case 1:
      document.getElementById("papersamples").value = ++paperSamples;
      break;
    case 2:
      document.getElementById("scissorssamples").value = ++scissorsSamples;
      break;
    case 3:
      document.getElementById("spocksamples").value = ++spockSamples;
      break;

    // Add a case for lizard samples.
    // HINT: Look at the previous cases.

    // YOUR CODE HERE
    case 4:
      document.getElementById("lizardsamples").value = ++lizardSamples;
      break;
  }
  label = parseInt(id);
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
      case 3:
        predictionText = "🖖";
        break;

      // Add a case for lizard samples.
      // HINT: Look at the previous cases.

      // YOUR CODE HERE
      case 4:
        predictionText = "🦎";
        break;
    }
    document.getElementById("prediction").value = predictionText;

    predictedClass.dispose();
    await tf.nextFrame();
  }
};

const doTraining = async () => {
  await train();
  console.info("Training Done!");
};

const startPredicting = () => {
  isPredicting = true;
  predict();
};

const stopPredicting = () => {
  isPredicting = false;
  predict();
};

const saveModel = () => {
  model.save("downloads://my_model");
};

// init
(async () => {
  await webcam.setup();
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
})();
