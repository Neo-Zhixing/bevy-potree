use crate::pipeline::{EyeDomeViewTarget, PointCloudBindGroup, PointCloudPipeline};
use crate::{PointCloudAsset, PointCloudDrawList, PointCloudUniform};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::DynamicUniformIndex;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{RenderLabel, ViewNode};
use bevy::render::render_resource::{
    LoadOp, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, ShaderStages, StoreOp,
};
use bevy::render::view::{ExtractedView, ViewDepthTexture, ViewTarget, ViewUniformOffset};

pub struct PointCloudNode {
    entity_query: QueryState<(
        &'static Handle<PointCloudAsset>,
        &'static DynamicUniformIndex<PointCloudUniform>,
    )>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PointCloudLabel;

impl FromWorld for PointCloudNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            entity_query: world.query_filtered(),
        }
    }
}

impl ViewNode for PointCloudNode {
    type ViewQuery = (
        &'static ExtractedView,
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static EyeDomeViewTarget,
        &'static PointCloudDrawList,
    );

    fn update(&mut self, world: &mut World) {
        self.entity_query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        (view, camera, target, depth, view_uniform_offset, eye_dome_view_target, draw_list): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let point_cloud_pipeline = world.resource::<PointCloudPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_assets = world.resource::<RenderAssets<PointCloudAsset>>();

        let mut tracked_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("point_cloud"),
            // NOTE: The opaque pass loads the color
            // buffer as well as writing to it.
            color_attachments: &[
                Some(target.get_color_attachment()),
                Some(RenderPassColorAttachment {
                    view: &eye_dome_view_target.depth_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK.into()),
                        store: StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth.view(),
                // NOTE: The opaque main pass loads the depth buffer and possibly overwrites it
                depth_ops: Some(Operations {
                    // NOTE: 0.0 is the far plane due to bevy's use of reverse-z projections.
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = camera.viewport.as_ref() {
            tracked_pass.set_camera_viewport(viewport);
        }

        let bind_groups = world.resource::<PointCloudBindGroup>();
        if bind_groups.bind_group.is_none() || bind_groups.model_bind_group.is_none() {
            return Ok(());
        }
        tracked_pass.set_bind_group(
            0,
            bind_groups.bind_group.as_ref().unwrap(),
            &[view_uniform_offset.offset],
        );
        tracked_pass.set_vertex_buffer(0, point_cloud_pipeline.instanced_point_quad.slice(0..32));
        for draw_data in &draw_list.list {
            let Some(pipeline) = pipeline_cache.get_render_pipeline(draw_data.pipeline_id) else {
                continue;
            };
            let Ok((point_cloud_asset, dynamic_index)) =
                self.entity_query.get_manual(world, draw_data.entity)
            else {
                continue;
            };
            let Some(point_cloud_asset) = render_assets.get(point_cloud_asset) else {
                continue;
            };

            tracked_pass.set_render_pipeline(pipeline);
            tracked_pass.set_bind_group(1, point_cloud_asset.bind_group.as_ref().unwrap(), &[]);
            tracked_pass.set_bind_group(
                2,
                bind_groups.model_bind_group.as_ref().unwrap(),
                &[dynamic_index.index()],
            );
            tracked_pass.draw(0..4, 0..point_cloud_asset.num_points);
        }
        drop(tracked_pass);

        let eye_dome_pipeline =
            pipeline_cache.get_render_pipeline(eye_dome_view_target.pipeline_id);
        if eye_dome_pipeline.is_none() {
            return Ok(());
        }
        let eye_dome_pipeline = eye_dome_pipeline.unwrap();

        let mut tracked_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("eye_dome_lighting"),
            // NOTE: The opaque pass loads the color
            // buffer as well as writing to it.
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = camera.viewport.as_ref() {
            tracked_pass.set_camera_viewport(viewport);
        }
        tracked_pass.set_render_pipeline(eye_dome_pipeline);

        let edl_strength: f32 = if view.projection.z_axis.w == -1.0 {
            // perspective projection
            // See https://github.com/bitshifter/glam-rs/blob/a35030d130c0464cbb07d6404df6843240182803/src/f32/scalar/mat4.rs#L843
            1.0
        } else {
            // orthographic projection
            // See https://github.com/bitshifter/glam-rs/blob/a35030d130c0464cbb07d6404df6843240182803/src/f32/scalar/mat4.rs#L924
            1.0 / view.projection.z_axis.z // near - far
        };

        tracked_pass.set_push_constants(
            ShaderStages::FRAGMENT,
            0,
            bytemuck::bytes_of(&edl_strength),
        );
        tracked_pass.set_bind_group(0, &eye_dome_view_target.bind_group, &[]);
        tracked_pass.set_vertex_buffer(0, point_cloud_pipeline.instanced_point_quad.slice(0..32));
        tracked_pass.draw(0..4, 0..1);
        Ok(())
    }
}
