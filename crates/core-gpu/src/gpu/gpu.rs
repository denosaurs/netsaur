use std::borrow::Cow;

use ndarray::{ArrayD, Dimension, IxDyn};

pub struct WGPUDataset {
    pub inputs: WGPUBuffer,
    pub outputs: WGPUBuffer,
}

pub struct WGPUBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub struct WGPUKernel {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    pub workgroups: (u32, u32, u32),
}

impl WGPUBackend {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ))
        .unwrap();

        Self { device, queue }
    }

    pub fn register(&mut self, source: String, workgroups: (u32, u32, u32)) -> WGPUKernel {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: "main",
            });

        let layout = pipeline.get_bind_group_layout(0);
        WGPUKernel {
            pipeline,
            layout,
            workgroups,
        }
    }

    pub fn execute(&mut self, kernel: &WGPUKernel, buffers: Vec<&WGPUBuffer>) {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer.buffer,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(buffer.size).unwrap()),
                }),
            })
            .collect();
        let bindgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.layout,
            entries: &entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bindgroup, &[]);
            pass.set_pipeline(&kernel.pipeline);
            let (group_x, group_y, group_z) = kernel.workgroups;
            pass.dispatch_workgroups(group_x, group_y, group_z);
        }
        self.queue.submit([encoder.finish()]);
    }
}

pub struct WGPUBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub shape: IxDyn,
}

impl WGPUBuffer {
    pub fn new(backend: &mut WGPUBackend, shape: IxDyn) -> Self {
        let buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: shape.size() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size: shape.size() as u64 * 4,
            shape,
        }
    }

    pub fn from(backend: &mut WGPUBackend, data: ArrayD<f32>) -> Self {
        let slice = data.as_slice().unwrap();
        let buffer = WGPUBuffer::new(backend, data.dim());
        let (_, bytes, _) = unsafe { slice.align_to() };
        backend.queue.write_buffer(&buffer.buffer, 0, bytes);
        buffer
    }

    pub fn read(&self, backend: &mut WGPUBackend) -> ArrayD<f32> {
        let buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, self.size);
        backend.queue.submit([encoder.finish()]);

        let (sender, receiver) = std::sync::mpsc::channel();
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
        backend.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let bytes = slice.get_mapped_range();
        let (_, data, _) = unsafe { bytes.align_to() };
        ArrayD::from_shape_vec(self.shape.clone(), data.to_vec()).unwrap()
    }

    pub fn write(&self, backend: &mut WGPUBackend, data: ArrayD<f32>) {
        let slice = data.as_slice().unwrap();
        let (_, bytes, _) = unsafe { slice.align_to() };
        backend.queue.write_buffer(&self.buffer, 0, bytes)
    }
}

impl Drop for WGPUBuffer {
    fn drop(&mut self) {
        self.buffer.destroy()
    }
}
