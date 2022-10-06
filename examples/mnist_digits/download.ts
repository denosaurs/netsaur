async function download(url: string, to: string) {
  console.log("Download", url);
  const f = await Deno.open(new URL(to, import.meta.url), {
    write: true,
    create: true,
  });
  await fetch(url).then((response) => {
    response.body!.pipeThrough(new DecompressionStream("gzip")).pipeTo(
      f.writable,
    );
  });
}

await download(
  "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
  "train-images.idx",
);
await download(
  "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
  "train-labels.idx",
);
await download(
  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
  "test-images.idx",
);
await download(
  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
  "test-labels.idx",
);
