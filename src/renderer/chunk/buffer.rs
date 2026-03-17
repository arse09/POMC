use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;
use azalea_core::position::ChunkPos;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

use super::mesher::ChunkMeshData;

pub const MAX_CHUNKS: usize = 8192;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkAABB {
    pub min: [f32; 4],
    pub max: [f32; 4],
}

struct ChunkMesh {
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    vertex_alloc: Allocation,
    index_alloc: Allocation,
    index_count: u32,
    aabb: ChunkAABB,
}

pub struct ChunkBufferStore {
    meshes: HashMap<ChunkPos, ChunkMesh>,
}

impl ChunkBufferStore {
    pub fn new(_device: &ash::Device, _allocator: &Arc<Mutex<Allocator>>) -> Self {
        Self {
            meshes: HashMap::new(),
        }
    }

    pub fn upload(
        &mut self,
        mesh: &ChunkMeshData,
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
    ) {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            self.remove(device, allocator, &mesh.pos);
            return;
        }

        self.remove(device, allocator, &mesh.pos);

        let vertex_bytes = bytemuck::cast_slice(&mesh.vertices);
        let index_bytes = bytemuck::cast_slice(&mesh.indices);

        let (vertex_buffer, vertex_alloc) = create_device_buffer(
            device,
            allocator,
            vertex_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );
        let (index_buffer, index_alloc) = create_device_buffer(
            device,
            allocator,
            index_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER,
        );

        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for v in &mesh.vertices {
            min_y = min_y.min(v.position[1]);
            max_y = max_y.max(v.position[1]);
        }

        let cx = mesh.pos.x as f32 * 16.0;
        let cz = mesh.pos.z as f32 * 16.0;

        self.meshes.insert(
            mesh.pos,
            ChunkMesh {
                vertex_buffer,
                index_buffer,
                vertex_alloc,
                index_alloc,
                index_count: mesh.indices.len() as u32,
                aabb: ChunkAABB {
                    min: [cx, min_y, cz, 0.0],
                    max: [cx + 16.0, max_y, cz + 16.0, 0.0],
                },
            },
        );
    }

    pub fn remove(
        &mut self,
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        pos: &ChunkPos,
    ) {
        if let Some(mesh) = self.meshes.remove(pos) {
            destroy_mesh(device, allocator, mesh);
        }
    }

    pub fn clear(&mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        for (_, mesh) in self.meshes.drain() {
            destroy_mesh(device, allocator, mesh);
        }
    }

    pub fn chunk_count(&self) -> u32 {
        self.meshes.len().min(MAX_CHUNKS) as u32
    }

    pub fn draw_all(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        for mesh in self.meshes.values() {
            unsafe {
                device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer], &[0]);
                device.cmd_bind_index_buffer(cmd, mesh.index_buffer, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
            }
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        self.clear(device, allocator);
    }
}

fn destroy_mesh(device: &ash::Device, allocator: &Arc<Mutex<Allocator>>, mesh: ChunkMesh) {
    let mut alloc = allocator.lock().unwrap();
    unsafe {
        device.destroy_buffer(mesh.vertex_buffer, None);
        device.destroy_buffer(mesh.index_buffer, None);
    }
    alloc.free(mesh.vertex_alloc).ok();
    alloc.free(mesh.index_alloc).ok();
}

fn create_device_buffer(
    device: &ash::Device,
    allocator: &Arc<Mutex<Allocator>>,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> (vk::Buffer, Allocation) {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(data.len() as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer =
        unsafe { device.create_buffer(&buffer_info, None) }.expect("failed to create buffer");
    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut allocation = allocator
        .lock()
        .unwrap()
        .allocate(&AllocationCreateDesc {
            name: "chunk_mesh",
            requirements: mem_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .expect("failed to allocate buffer memory");

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .expect("failed to bind buffer memory");
    }

    allocation.mapped_slice_mut().unwrap()[..data.len()].copy_from_slice(data);

    (buffer, allocation)
}
