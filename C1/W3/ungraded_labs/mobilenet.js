(async () => {
  const img = document.getElementById("img");
  const outp = document.getElementById("output");
  const model = await mobilenet.load();
  const predictions = await model.classify(img);

  console.log(JSON.stringify(predictions, null, "\t"));
  predictions.forEach((pred) => {
    outp.innerHTML += "<br/>" + pred.className + " : " + pred.probability;
  });
})();
