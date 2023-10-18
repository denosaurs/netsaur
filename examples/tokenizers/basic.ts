import { init, Tokenizer } from "../../tokenizers/mod.ts";

await init();

const tokenizer = Tokenizer.fromJSON(
  await (await fetch(
    `https://huggingface.co/satvikag/chatbot/resolve/main/tokenizer.json`,
  )).text(),
);

console.log(tokenizer.tokenize("Hello World!"));