import { Shape } from "./backend.ts";
export class WebGPUBackend {
  public adapter: GPUAdapter;
  public device: GPUDevice;
  public shaderModule: GPUShaderModule;

  public bindGroupLayout: GPUBindGroupLayout;

  public pipeline: GPUComputePipeline;

  static async init() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter?.requestDevice();

    if (!device || !adapter) {
      throw new Error("no suitable device/adapter found");
    }

    return new WebGPUBackend(adapter, device);
  }

  constructor(adapter: GPUAdapter, device: GPUDevice) {
    this.adapter = adapter;
    this.device = device;
    this.shaderModule = this.device.createShaderModule({
      code: Deno.readTextFileSync(new URL("./matmul.wgsl", import.meta.url)),
    });
    this.pipeline = this.device.createComputePipeline({
      compute: { module: this.shaderModule, entryPoint: "main" },
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }

  public async matMul(
    a: number[],
    b: number[],
    shapeA: Shape,
    shapeB: Shape,
  ): Promise<Float32Array> {
    const matA = new Float32Array([...shapeA, ...a]);
    const matB = new Float32Array([...shapeB, ...b]);

    const bufA = this.device.createBuffer({
      size: matA.byteLength,
      mappedAtCreation: true,
      usage: GPUBufferUsage.STORAGE,
    });
    new Float32Array(bufA.getMappedRange()).set(matA);
    bufA.unmap();

    const bufB = this.device.createBuffer({
      size: matB.byteLength,
      mappedAtCreation: true,
      usage: GPUBufferUsage.STORAGE,
    });
    new Float32Array(bufB.getMappedRange()).set(matB);
    bufB.unmap();

    const resultSize = (2 + shapeA[0] * shapeB[1]) *
      Float32Array.BYTES_PER_ELEMENT;
    const bufResult = this.device.createBuffer({
      size: resultSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const bufRead = this.device.createBuffer({
      size: resultSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 2,
          resource: {
            buffer: bufResult,
          },
        },
        {
          binding: 0,
          resource: {
            buffer: bufA,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: bufB,
          },
        },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatch(shapeA[0], shapeB[1]);
    computePass.endPass();

    commandEncoder.copyBufferToBuffer(bufResult, 0, bufRead, 0, resultSize);

    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);

    await bufRead.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(bufRead.getMappedRange().slice(0));
    bufRead.unmap();

    bufA.destroy();
    bufB.destroy();
    bufResult.destroy();
    bufRead.destroy();

    return res.slice(2);
  }
}
