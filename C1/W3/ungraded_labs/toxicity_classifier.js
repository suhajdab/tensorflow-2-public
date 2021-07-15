(async () => {
  const threshold = 0.9;
  const model = await toxicity.load(threshold);

  // allow user input for sentences
  const sentences = (
    await prompt(
      "Semicolon separated phrases",
      "You are awesome; You worthless piece of stinky trash; I'm gonna kill you all; You sucks my shoelace; I'll fuck you up"
    )
  )
    .split(";")
    .map((s) => s.trim())
    .filter((s) => s !== "");

  const predictions = await model.classify(sentences);

  sentences.forEach((sentence, i) => {
    // format results per sentence
    // ex: You worthless piece of stinky trash {insult: 0.993285059928894} {toxicity: 0.9926277995109558}
    const current = predictions
      .filter((pred) => pred.results[i].match === true)
      .map((pred) => ({ [pred.label]: pred.results[i].probabilities[1] }));
    if (!current.length) current.push("🤷");
    console.log(sentence, ...current);
  });
  // console.log(predictions);
})();
