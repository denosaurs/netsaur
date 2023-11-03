use std::borrow::Cow;

pub struct WGPUBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub kernels: Vec<WGPUKernel>,
}

pub struct WGPUKernel {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

impl WGPUBackend {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::empty(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
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

        Self {
            device,
            queue,
            kernels: Vec::new(),
        }
    }

    pub fn register(&mut self, source: Cow<'_, str>) {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source),
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
        self.kernels.push(WGPUKernel { pipeline, layout });
    }

    pub fn execute(&mut self, kernel: usize, buffers: Vec<WGPUBuffer>) {
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
            layout: &self.kernels[kernel].layout,
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
            pass.set_pipeline(&self.kernels[kernel].pipeline);
            pass.dispatch_workgroups(8, 8, 8);
        }
        self.queue.submit([encoder.finish()]);
    }
}

pub struct WGPUBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
}

impl WGPUBuffer {
    pub fn new(backend: &mut WGPUBackend, size: u64) -> Self {
        let buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self { buffer, size }
    }

    pub fn read(&self, backend: &mut WGPUBackend) -> Vec<u8> {
        let buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC,
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
        let data = slice.get_mapped_range();

        data.to_vec()
    }

    pub fn write(&self, backend: &mut WGPUBackend, data: &[u8]) {
        backend.queue.write_buffer(&self.buffer, 0, data)
    }
}

impl Drop for WGPUBuffer {
    fn drop(&mut self) {
        self.buffer.destroy()
    }
}
