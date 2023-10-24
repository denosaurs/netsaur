import { init, Tokenizer } from "../../tokenizers/mod.ts";

await init();

const tokenizer = Tokenizer.fromJSON(
  await (await fetch(
    `https://huggingface.co/satvikag/chatbot/resolve/main/tokenizer.json`,
  )).text(),
);

const encoded = tokenizer.encode("Hello World!");
console.log(encoded);
const decoded = tokenizer.decode(encoded);
console.log(decoded);