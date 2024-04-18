mod clippling_planes;
#[cfg(feature = "las")]
mod las_loader;
#[cfg(feature = "opd")]
mod opd_loader;
mod pipeline;
mod playback;
mod point_cloud;
mod render;
mod render_graph;
use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        extract_component::UniformComponentPlugin,
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssetPlugin,
        render_graph::{RenderGraphApp, ViewNodeRunner},
        render_resource::{ShaderStage, SpecializedRenderPipelines},
        view::{check_visibility, VisibilitySystems},
        Render, RenderApp, RenderSet,
    },
};
pub use clippling_planes::{ClippingPlaneBundle, ClippingPlaneRange};
#[cfg(feature = "las")]
pub use las_loader::*;
#[cfg(feature = "opd")]
pub use opd_loader::*;
pub use pipeline::*;
pub use playback::*;
pub use point_cloud::*;
pub use render::*;
pub use render_graph::*;

#[derive(Default)]
pub struct PointCloudPlugin;

impl Plugin for PointCloudPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_asset::<PointCloudAsset>();

        #[cfg(feature = "las")]
        app.init_asset_loader::<LasLoader>();
        #[cfg(feature = "opd")]
        app.init_asset_loader::<OpdLoader>();

        app.add_plugins((
            RenderAssetPlugin::<PreparedPointCloudAsset>::default(),
            UniformComponentPlugin::<PointCloudUniform>::default(),
            ExtractResourcePlugin::<PointCloudPlaybackControls>::default(),
        ))
        .add_systems(
            PostUpdate,
            (
                PointCloudPlaybackControls::playback_system,
                check_visibility::<With<PotreePointCloud>>
                    .in_set(VisibilitySystems::CheckVisibility),
            ),
        )
        .init_resource::<PointCloudPlaybackControls>();

        load_internal_asset!(
            app,
            POINT_CLOUD_VERT_SHADER_HANDLE,
            "shader.vert",
            |s, path| { Shader::from_glsl(s, ShaderStage::Vertex, path) }
        );
        load_internal_asset!(
            app,
            POINT_CLOUD_FRAG_SHADER_HANDLE,
            "shader.frag",
            |s, path| { Shader::from_glsl(s, ShaderStage::Fragment, path) }
        );
        load_internal_asset!(
            app,
            EYE_DOME_LIGHTING_SHADER_HANDLE,
            "eye-dome.wgsl",
            Shader::from_wgsl
        );
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_systems(
                ExtractSchedule,
                (
                    extract_point_cloud,
                    clippling_planes::extract_clipping_planes,
                ),
            )
            .add_systems(
                Render,
                (clippling_planes::prepare_clipping_planes,).in_set(RenderSet::Prepare),
            )
            .add_systems(
                Render,
                (
                    queue_point_cloud_bind_group,
                    queue_view_targets,
                    queue_point_cloud,
                )
                    .in_set(RenderSet::Queue),
            )
            .init_resource::<clippling_planes::UniformBufferOfGpuClippingPlaneRanges>()
            .init_resource::<PointCloudBindGroup>();

        render_app
            .add_systems(Render, prepare_animated_assets.in_set(RenderSet::Prepare))
            .init_resource::<PointCloudPlaybackControls>();

        render_app
            .add_render_graph_node::<ViewNodeRunner<PointCloudNode>>(Core3d, PointCloudLabel)
            .add_render_graph_edge(Core3d, Node3d::EndMainPass, PointCloudLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<PointCloudPipeline>()
            .init_resource::<SpecializedRenderPipelines<PointCloudPipeline>>()
            .init_resource::<EyeDomePipeline>()
            .init_resource::<SpecializedRenderPipelines<EyeDomePipeline>>();
    }
}
