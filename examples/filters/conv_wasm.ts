import {
    Activation,
    Conv2DLayer,
    Cost,
    WASM,
    Rank,
    Sequential,
    setupBackend,
    Tensor,
    tensor4D,
  } from "../../mod.ts";
  import { decode } from "https://deno.land/x/pngs@0.1.1/mod.ts";
  import { createCanvas } from "https://deno.land/x/canvas@v1.4.1/mod.ts";
  
  const canvas = createCanvas(600, 600);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, 600, 600);
  
  const dim = 28;
  const kernel = [[[
    [-1, 1, 0],
    [-1, 1, 0],
    [-1, 1, 0],
  ]]];
  
  //Credit: Hashrock (https://github.com/hashrock)
  const img = decode(Deno.readFileSync("./examples/filters/deno.png")).image;
  const buf = new Float32Array(dim * dim);
  for (let i = 0; i < dim * dim; i++) {
    buf[i] = img[i * 4];
  }
  
  await setupBackend(WASM);
  
  const net = new Sequential({
    size: [1, 1, dim, dim],
    silent: true,
    layers: [
      Conv2DLayer({
        activation: Activation.Linear,
        kernel: tensor4D(kernel),
        kernelSize: [1, 1, 3, 3],
        padding: 1,
        strides: [1, 1],
        unbiased: true,
      }),
      // PoolLayer({ strides: [2, 2], mode: PoolMode.Max }),
    ],
    cost: Cost.MSE,
  });
  
  const data = new Tensor(buf, [1, 1, dim, dim]);
  const result = await net.predict(data) as Tensor<Rank.R4>;
  
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      const pixel = buf[j * dim + i];
      ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
      ctx.fillRect(i * 10, j * 10, 10, 10);
    }
  }
  
  for (let i = 0; i < result.shape[1]; i++) {
    for (let j = 0; j < result.shape[2]; j++) {
      const pixel = result.data[j * result.shape[2] + i]
      ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
      ctx.fillRect(i * 10 + dim * 10, j * 10, 10, 10);
    }
  }
  
  // for (let i = 0; i < pool.output.x; i++) {
  //   for (let j = 0; j < pool.output.y; j++) {
  //     const pixel = Math.round(
  //       Math.max(Math.min(pool.output.data[j * pool.output.x + i], 255), 0),
  //     );
  //     ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
  //     ctx.fillRect(i * 20 + dim * 10, j * 20 + dim * 10, 20, 20);
  //   }
  // }
  
  await Deno.writeFile("./examples/filters/output.png", canvas.toBuffer());
  