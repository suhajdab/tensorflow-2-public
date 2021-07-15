(async () => {
  const MODEL_URL = "output_files/model.json"; // point to github version, as jupyter notebook failed to run
  const model = await tf.loadLayersModel(MODEL_URL);
  model.summary();
  const input = tf.tensor2d([10.0], [1, 1]);
  const result = model.predict(input);
  console.log(`%c 👏 ${result.dataSync()}`, "color: #f60; font-size: 14px");
})();
