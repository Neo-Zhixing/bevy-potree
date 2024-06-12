use crate::{pipeline::PointCloudPipeline, PointCloudAsset};
use crate::{PointCloudPipelineKey, ATTRIBUTE_COLOR};
use bevy::render::render_asset::{RenderAssetUsages, RenderAssets};
use bevy::render::render_resource::{
    BufferDescriptor, CachedRenderPipelineId, DynamicBindGroupEntries, PipelineCache,
    SpecializedRenderPipelines,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::view::VisibleEntities;
use bevy::{
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    prelude::*,
    render::{
        render_asset::RenderAsset,
        render_resource::{BindGroup, Buffer, BufferInitDescriptor, BufferUsages, ShaderType},
        renderer::RenderDevice,
        Extract,
    },
};
use opd_parser::Frames;
#[derive(Component, Clone)]
pub struct PotreePointCloud {
    pub mesh: Handle<PointCloudAsset>,
    pub point_size: f32,
}
#[derive(Component, Clone, ShaderType)]
pub struct PointCloudUniform {
    pub transform: Mat4,
    pub point_size: f32,
}

pub(crate) fn extract_point_cloud(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    query: Extract<Query<(Entity, &PotreePointCloud, &GlobalTransform)>>,
) {
    let mut values = Vec::with_capacity(*previous_len);

    for (entity, point_cloud, transform) in query.iter() {
        values.push((
            entity,
            (
                PointCloudUniform {
                    transform: transform.compute_matrix(),
                    point_size: point_cloud.point_size,
                },
                point_cloud.mesh.clone(),
            ),
        ));
    }
    *previous_len = values.len();
    commands.insert_or_spawn_batch(values);
}

#[derive(Component)]
pub struct PointCloudDrawList {
    pub list: Vec<PointCloudDrawData>,
}

pub struct PointCloudDrawData {
    pub entity: Entity,
    pub pipeline_id: CachedRenderPipelineId,
}

pub(crate) fn queue_point_cloud(
    pipeline: Res<PointCloudPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<PointCloudPipeline>>,
    cache: Res<PipelineCache>,
    views: Query<(Entity, &VisibleEntities)>,
    items: Query<&Handle<PointCloudAsset>>,
    point_clouds: Res<RenderAssets<PointCloudAsset>>,
    msaa: Option<Res<Msaa>>,
    mut commands: Commands,
) {
    let msaa = msaa.map(|a| a.samples()).unwrap_or(1);
    for (view_entity, entities) in &views {
        let mut list = vec![];
        for &entity in &entities.entities {
            if let Some(asset) = items
                .get(entity)
                .ok()
                .and_then(|handle| point_clouds.get(handle))
            {
                let key = PointCloudPipelineKey {
                    colored: asset.colored,
                    animated: asset.animation_buffer.is_some(),
                    msaa,
                };

                let pipeline_id = pipelines.specialize(&cache, &pipeline, key);
                list.push(PointCloudDrawData {
                    entity,
                    pipeline_id,
                });
            }
        }
        if !list.is_empty() {
            commands
                .entity(view_entity)
                .insert(PointCloudDrawList { list });
        }
    }
}

pub struct PreparedPointCloudAsset {
    pub buffer: Buffer,
    pub num_points: u32,
    pub bind_group: Option<BindGroup>,

    pub animation_buffer: Option<(Buffer, Buffer)>,
    pub frames: Option<Frames>,
    pub current_animation_frame: usize,
    pub animation_time: f32,
    pub animation_frame_start_time: f32,
    pub animation_scale: Vec3,

    pub colored: bool,
}

impl PreparedPointCloudAsset {
    pub fn seek(
        &mut self,
        seek_to: f32, // time from the start of the animation to seek to
        queue: &RenderQueue,
        render_device: &RenderDevice,
        pipeline: &PointCloudPipeline,
    ) {
        let (prev_animation_buffer, next_animation_buffer) = self.animation_buffer.as_mut().expect(
            "Cannot call PreparedPointCloudAsset::seek on an instance without an animation",
        );
        let frames = match self.frames.as_ref().unwrap() {
            Frames::I8(frames) => frames,
            _ => todo!(), // make some kinda trait abstraction
        };

        self.animation_time = seek_to;

        // If we're already in the correct frame, adjust interpolation and exit
        let current_frame_end_time = frames[self.current_animation_frame].time / 1000.;
        if (self.animation_frame_start_time..current_frame_end_time).contains(&self.animation_time)
        {
            let duration = current_frame_end_time - self.animation_frame_start_time;
            let delta = self.animation_time - self.animation_frame_start_time;
            let interpolation = (delta / duration).min(1.0);

            queue.write_buffer(next_animation_buffer, 0, bytemuck::bytes_of(&interpolation));

            return;
        }

        let to_enter = if self.current_animation_frame == frames.len() - 1 && {
            // if we're on the last frame, check if we're seeking to the first frame first
            let first_frame_end_time = frames[0].time / 1000.;
            self.animation_time < first_frame_end_time
        } {
            0
        } else if self.current_animation_frame != frames.len() - 1 && {
            // if we're not on the last frame, check if we're seeking to the next frame first
            let next_frame_end_time = frames[self.current_animation_frame + 1].time / 1000.;
            (current_frame_end_time..next_frame_end_time).contains(&self.animation_time)
        } {
            self.current_animation_frame + 1
        } else {
            // if we're seeking to neither, find what frame we're seeking to using binary search
            match frames.binary_search_by(|f| (f.time / 1000.).partial_cmp(&seek_to).unwrap()) {
                // Landed on the exact end of a frame
                // If its the last frame, stay on the last frame
                // If its not, we will enter the next frame
                Ok(index) if index == frames.len() - 1 => index,
                Ok(index) => index + 1,
                // The index is to the frame where `delta_seconds` is smaller than the indexed frame, but larger than the previous,
                // or its out of bounds
                Err(index) if index == frames.len() => panic!("Out of bounds seek"),
                Err(index) => index,
            }
        };

        let mut view = vec![0.0; self.num_points as usize * 3];

        if to_enter == self.current_animation_frame + 1 {
            // We're going to the next frame, so the start time is simply the current end time
            self.animation_frame_start_time = current_frame_end_time;
        } else {
            // We're not going to the next frame, so the current next_animation_buffer cannot
            // be used as the upcoming prev_animation_buffer directly.
            // Adjust the next_animation_buffer to point to the frame before the one we're entering
            // or zero it out if we're entering the first frame

            if to_enter == 0 {
                // We're entering the first frame, set the start time to zero
                self.animation_frame_start_time = 0.;
            } else {
                // We're not entering the first frame, setup `view` with the values in frame `to_enter - 1`
                // Also set the frame start time
                for (i, arr) in frames[to_enter - 1].into_iter().enumerate() {
                    let arr = Vec3::from(arr) * self.animation_scale;
                    for j in 0..3 {
                        view[i * 3 + j] = arr[j];
                    }
                }
                self.animation_frame_start_time = frames[to_enter - 1].time / 1000.;
            }

            // Write the buffer
            queue.write_buffer(next_animation_buffer, 4, bytemuck::cast_slice(&view));
        }

        // Swap the buffers
        std::mem::swap(next_animation_buffer, prev_animation_buffer);

        // If we're moving to the previous frame, the values in the next_animation_buffer are
        // already set up for it thanks to the swap, so we can skip this step
        if to_enter + 1 != self.current_animation_frame {
            // Setup view with values for the frame we're entering
            for (i, arr) in frames[to_enter].into_iter().enumerate() {
                let arr = Vec3::from(arr) * self.animation_scale;
                for j in 0..3 {
                    view[i * 3 + j] = arr[j];
                }
            }

            // Write the values into next_animation_buffer
            queue.write_buffer(next_animation_buffer, 4, bytemuck::cast_slice(&view));
        }

        self.current_animation_frame = to_enter;

        // Calculate and write interpolation for the frame we just entered
        let current_frame_end_time = frames[self.current_animation_frame].time / 1000.;
        let duration = current_frame_end_time - self.animation_frame_start_time;
        let delta = self.animation_time - self.animation_frame_start_time;
        let interpolation = (delta / duration).min(1.0);

        queue.write_buffer(next_animation_buffer, 0, bytemuck::bytes_of(&interpolation));

        // Update the bind group, since we swapped the buffers.
        self.update_bind_group(render_device, pipeline);
    }

    pub fn update_bind_group(
        &mut self,
        render_device: &RenderDevice,
        pipeline: &PointCloudPipeline,
    ) {
        let mut bind_group_entries =
            DynamicBindGroupEntries::sequential((self.buffer.as_entire_binding(),));
        if let Some((animation_buffer, next)) = self.animation_buffer.as_ref() {
            bind_group_entries = bind_group_entries.extend_sequential((
                animation_buffer.as_entire_binding(),
                next.as_entire_binding(),
            ));
        }
        let bind_group = render_device.create_bind_group(
            "point cloud buffer bind group",
            if self.animation_buffer.is_some() {
                &pipeline.animated_entity_layout
            } else {
                &pipeline.entity_layout
            },
            &bind_group_entries,
        );
        self.bind_group = Some(bind_group);
    }
}

impl RenderAsset for PointCloudAsset {
    type PreparedAsset = PreparedPointCloudAsset;

    type Param = (SRes<RenderDevice>, SRes<PointCloudPipeline>);

    fn asset_usage(&self) -> RenderAssetUsages {
        RenderAssetUsages::default()
    }

    fn prepare_asset(
        self,
        (render_device, pipeline): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, bevy::render::render_asset::PrepareAssetError<Self>> {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            usage: BufferUsages::STORAGE,
            label: Some("Point cloud vertex buffer"),
            contents: self.mesh.get_vertex_buffer_data().as_slice(),
        });

        let animation_buffer = if self.animation.is_some() {
            let size = self.mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len() as u64
                * std::mem::size_of::<f32>() as u64
                * 3
                + std::mem::size_of::<f32>() as u64;
            let animation_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("AnimationBuffer"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let animation_buffer_next = render_device.create_buffer(&BufferDescriptor {
                label: Some("AnimationBufferNext"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Some((animation_buffer, animation_buffer_next))
        } else {
            None
        };
        let mut asset = PreparedPointCloudAsset {
            buffer,
            num_points: self.mesh.count_vertices() as u32,
            bind_group: None,
            animation_buffer,
            frames: self.animation,
            current_animation_frame: 0,
            animation_time: 0.0,
            animation_frame_start_time: 0.0,
            animation_scale: self.animation_scale,
            colored: self.mesh.contains_attribute(ATTRIBUTE_COLOR),
        };
        asset.update_bind_group(render_device, pipeline);
        Ok(asset)
    }
}
