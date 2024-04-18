use bevy::{
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::ComponentUniforms,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        texture::{BevyDefault, TextureCache},
        view::{ExtractedView, ViewTarget, ViewUniforms},
    },
    utils::HashMap,
};

use crate::{
    clippling_planes::UniformBufferOfGpuClippingPlaneRanges, PointCloudPlaybackControls,
    PointCloudUniform, PreparedPointCloudAsset,
};

pub(crate) const POINT_CLOUD_VERT_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0x3fc9d1ff70cedf01);
pub(crate) const POINT_CLOUD_FRAG_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0x3fc9d1ff70cedf02);
pub(crate) const EYE_DOME_LIGHTING_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0x3fc9d1ff70cedf03);

#[derive(Resource)]
pub struct PointCloudPipeline {
    pub view_layout: BindGroupLayout,
    pub entity_layout: BindGroupLayout,
    pub animated_entity_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,

    pub instanced_point_quad: Buffer,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct PointCloudPipelineKey {
    pub colored: bool,
    pub animated: bool,
    pub msaa: u32,
}

#[derive(Resource)]
pub struct EyeDomePipeline {
    pub eye_dome_image_layout: BindGroupLayout,
    pub multisampled_eye_dome_image_layout: BindGroupLayout,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct EyeDomePipelineKey {
    pub msaa: u32,
}

#[derive(Resource, Default)]
pub struct PointCloudBindGroup {
    pub bind_group: Option<BindGroup>,
    pub model_bind_group: Option<BindGroup>,
}
pub(crate) fn queue_point_cloud_bind_group(
    render_device: Res<RenderDevice>,
    pipeline: Res<PointCloudPipeline>,
    view_uniform: Res<ViewUniforms>,
    clipping_planes_uniform: Res<UniformBufferOfGpuClippingPlaneRanges>,
    model_uniform: Res<ComponentUniforms<PointCloudUniform>>,
    mut bind_groups: ResMut<PointCloudBindGroup>,
) {
    if let (Some(view_uniform_resource), Some(clipping_plane_resource)) = (
        view_uniform.uniforms.binding(),
        clipping_planes_uniform.0.binding(),
    ) {
        let bind_group = render_device.create_bind_group(
            "point_cloud_bind_group",
            &pipeline.view_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: view_uniform_resource,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: clipping_plane_resource,
                },
            ],
        );
        bind_groups.bind_group = Some(bind_group);
    }

    if let Some(binding) = model_uniform.uniforms().binding() {
        bind_groups.model_bind_group = Some(render_device.create_bind_group(
            "point_cloud_model_bind_group",
            &pipeline.model_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: binding,
            }],
        ));
    }
}

// this is not just used for the points, it doubles as a fullscreen quad. do not change
const QUAD_VERTEX_BUF: &[f32] = &[0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

impl FromWorld for PointCloudPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let instanced_point_quad = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: "instanced point quad".into(),
            contents: bytemuck::cast_slice(QUAD_VERTEX_BUF),
            usage: BufferUsages::VERTEX,
        });
        let view_layout = render_device.create_bind_group_layout(
            Some("PointCloudViewLabel"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );
        let entity_layout = render_device.create_bind_group_layout(
            Some("PointCloudViewLayout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        );
        let animated_entity_layout = render_device.create_bind_group_layout(
            Some("PointCloudViewLayout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );
        let model_layout = render_device.create_bind_group_layout(
            Some("PointCloudModelLayout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        );

        Self {
            view_layout,
            model_layout,
            entity_layout,
            animated_entity_layout,
            instanced_point_quad,
        }
    }
}

impl SpecializedRenderPipeline for PointCloudPipeline {
    type Key = PointCloudPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let PointCloudPipelineKey {
            colored,
            animated,
            msaa,
        } = key;

        RenderPipelineDescriptor {
            label: Some("point_cloud_pipeline".into()),
            layout: vec![
                self.view_layout.clone(),
                if animated {
                    self.animated_entity_layout.clone()
                } else {
                    self.entity_layout.clone()
                },
                self.model_layout.clone(),
            ],
            vertex: VertexState {
                shader: POINT_CLOUD_VERT_SHADER_HANDLE,
                shader_defs: {
                    let mut defs = Vec::new();
                    if colored {
                        defs.push("COLORED".into());
                    }
                    if animated {
                        defs.push("ANIMATED".into());
                    }
                    defs
                },
                entry_point: "main".into(),
                buffers: vec![VertexBufferLayout {
                    array_stride: 8,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(FragmentState {
                shader: POINT_CLOUD_FRAG_SHADER_HANDLE,
                shader_defs: {
                    let mut defs = Vec::new();
                    if colored {
                        defs.push("COLORED".into());
                    }
                    if animated {
                        defs.push("ANIMATED".into());
                    }
                    defs
                },
                entry_point: "main".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba8UnormSrgb,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::R32Float,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::RED,
                    }),
                ],
            }),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: msaa,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: default(),
        }
    }
}

impl FromWorld for EyeDomePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let eye_dome_image_layout = render_device.create_bind_group_layout(
            Some("EyeDomeImageLayout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        );

        let multisampled_eye_dome_image_layout = render_device.create_bind_group_layout(
            Some("MultisampledEyeDomeImageLayout"),
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: true,
                },
                count: None,
            }],
        );

        Self {
            eye_dome_image_layout,
            multisampled_eye_dome_image_layout,
        }
    }
}

impl SpecializedRenderPipeline for EyeDomePipeline {
    type Key = EyeDomePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let msaa = key.msaa;

        RenderPipelineDescriptor {
            label: Some("EyeDomeLightingPipeline".into()),
            layout: vec![if msaa > 1 {
                self.multisampled_eye_dome_image_layout.clone()
            } else {
                self.eye_dome_image_layout.clone()
            }],
            vertex: VertexState {
                shader: EYE_DOME_LIGHTING_SHADER_HANDLE,
                shader_defs: if msaa > 1 {
                    vec!["MULTISAMPLED".into()]
                } else {
                    default()
                },
                entry_point: "vertex".into(),
                buffers: vec![VertexBufferLayout {
                    array_stride: 8,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: msaa,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: EYE_DOME_LIGHTING_SHADER_HANDLE,
                shader_defs: if msaa > 1 {
                    vec!["MULTISAMPLED".into()]
                } else {
                    default()
                },
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::SrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::Zero,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                range: 0..std::mem::size_of::<f32>() as u32,
            }],
        }
    }
}

#[derive(Clone, Component)]
pub struct EyeDomeViewTarget {
    pub depth_texture: Texture,
    pub depth_texture_view: TextureView,
    pub bind_group: BindGroup,
    pub pipeline_id: CachedRenderPipelineId,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_view_targets(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    pipeline_cache: Res<PipelineCache>,
    eye_dome_pipeline: Res<EyeDomePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<EyeDomePipeline>>,
    cameras: Query<(Entity, &ExtractedCamera, &ExtractedView, &ViewTarget)>,
    msaa: Option<Res<Msaa>>,
) {
    let msaa = msaa.map(|a| a.samples()).unwrap_or(1);
    let mut textures = HashMap::default();

    for (entity, camera, _view, _view_target) in cameras.iter() {
        if let Some(target_size) = camera.physical_target_size {
            let size = Extent3d {
                width: target_size.x,
                height: target_size.y,
                depth_or_array_layers: 1,
            };

            let main_textures = textures.entry(camera.target.clone()).or_insert_with(|| {
                let depth_descriptor = TextureDescriptor {
                    label: None,
                    size,
                    mip_level_count: 1,
                    sample_count: msaa,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Float,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                };
                let cached_depth_texture = texture_cache.get(&render_device, depth_descriptor);

                let bind_group = render_device.create_bind_group(
                    "eye_dome_bind_group",
                    if msaa > 1 {
                        &eye_dome_pipeline.multisampled_eye_dome_image_layout
                    } else {
                        &eye_dome_pipeline.eye_dome_image_layout
                    },
                    &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&cached_depth_texture.default_view),
                    }],
                );
                EyeDomeViewTarget {
                    depth_texture: cached_depth_texture.texture,
                    depth_texture_view: cached_depth_texture.default_view,
                    bind_group,
                    pipeline_id: pipelines.specialize(
                        &pipeline_cache,
                        &eye_dome_pipeline,
                        EyeDomePipelineKey { msaa },
                    ),
                }
            });

            commands.entity(entity).insert(main_textures.clone());
        }
    }
}

impl ExtractResource for PointCloudPlaybackControls {
    type Source = Self;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

pub fn prepare_animated_assets(
    queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
    pipeline: Res<PointCloudPipeline>,
    mut assets: ResMut<RenderAssets<PreparedPointCloudAsset>>,
    playback: ResMut<PointCloudPlaybackControls>,
) {
    for (handle, asset) in assets.iter_mut() {
        if asset.animation_buffer.is_some() {
            let playback = playback
                .controls
                .get(&Handle::Weak(handle))
                .copied()
                .unwrap_or_default();
            if playback.time != asset.animation_time {
                asset.seek(playback.time, &queue, &render_device, &pipeline);
            }
        }
    }
}
