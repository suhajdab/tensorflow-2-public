const hotEncodeLabels = ({ xs, ys }) => {
  const labels = [
    ys.species == "setosa" ? 1 : 0,
    ys.species == "virginica" ? 1 : 0,
    ys.species == "versicolor" ? 1 : 0,
  ];
  return { xs: Object.values(xs), ys: Object.values(labels) };
};

const getTestCase = (n) => {
  // Test Cases:

  // Setosa
  // const testVal = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);

  // Versicolor
  // const testVal = tf.tensor2d([6.4, 3.2, 4.5, 1.5], [1, 4]);

  // Virginica
  // const testVal = tf.tensor2d([5.8,2.7,5.1,1.9], [1, 4]);
  const testCases = [
    {
      data: [4.4, 2.9, 1.4, 0.2],
      label: 0,
    },
    {
      data: [6.4, 3.2, 4.5, 1.5],
      label: 2,
    },
    {
      data: [5.8, 2.7, 5.1, 1.9],
      label: 1,
    },
  ];

  const testCase = testCases[n];
  const testData = tf.tensor2d(testCase.data, [1, testCase.data.length]);

  return { testData, label: testCase.label };
};

const getData = () => {
  const csvUrl = "./data/iris.csv";
  return tf.data.csv(csvUrl, {
    columnConfigs: {
      // "species" comes from column name
      species: {
        isLabel: true,
      },
    },
  });
};

const train = async (data) => {
  const numOfFeatures = (await data.columnNames()).length - 1;
  const convertedData = data.map(hotEncodeLabels).batch(10);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [numOfFeatures],
      activation: "sigmoid",
      units: 5,
    })
  );
  model.add(tf.layers.dense({ activation: "softmax", units: 3 }));

  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.06),
  });

  const onEpochEnd = async (epoch, logs) => {
    console.log("Epoch: " + epoch + " Loss: " + logs.loss);
  };

  await model.fitDataset(convertedData, {
    epochs: 100,
    callbacks: { onEpochEnd },
  });
  return model;
};

const predict = async (model) => {
  for (i = 0; i < 3; i++) {
    const { testData, label } = getTestCase(i);

    const prediction = model.predict(testData);
    const pIndex = tf.argMax(prediction, (axis = 1)).dataSync();

    const classNames = ["Setosa", "Virginica", "Versicolor"];

    console.log(prediction.dataSync());
    console.log(
      `%c prediction = ${classNames[pIndex]}, expected = ${classNames[label]}`,
      "color: #f60; font-size: 12px"
    );
  }
};

const run = async () => {
  try {
    const data = getData();
    const model = await train(data);
    await predict(model);
  } catch (e) {
    console.error(e);
  }
};

run();
